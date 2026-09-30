// DuckDB-WASM: owns the data from upload to results.
import * as duckdb from "@duckdb/duckdb-wasm";
import ehWasm from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import ehWorker from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import mvpWasm from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import mvpWorker from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import type { Table } from "apache-arrow";

let db: duckdb.AsyncDuckDB | null = null;
let conn: duckdb.AsyncDuckDBConnection | null = null;

export async function connect(): Promise<duckdb.AsyncDuckDBConnection> {
  if (conn) return conn;
  const bundle = await duckdb.selectBundle({
    mvp: { mainModule: mvpWasm, mainWorker: mvpWorker },
    eh: { mainModule: ehWasm, mainWorker: ehWorker },
  });
  db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), new Worker(bundle.mainWorker!));
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  conn = await db.connect();
  return conn;
}

export async function query(sql: string): Promise<Table> {
  return (await connect()).query(sql);
}

export const ident = (name: string) => `"${name.replace(/"/g, '""')}"`;
const literal = (text: string) => `'${text.replace(/'/g, "''")}'`;

/** Load a CSV or Parquet file (user upload or fetched sample) as table `source`. */
export async function loadFile(name: string, data: File | Uint8Array): Promise<void> {
  await connect();
  const path = `upload_${Date.now()}_${name.replace(/[^\w.-]/g, "_")}`;
  if (data instanceof Uint8Array) await db!.registerFileBuffer(path, data);
  else await db!.registerFileHandle(path, data, duckdb.DuckDBDataProtocol.BROWSER_FILEREADER, true);
  const reader = /\.parquet$/i.test(name) ? `read_parquet(${literal(path)})` : // No BOOLEAN guessing: "yes"/"no" labels stay as written, as pandas reads them.
      `read_csv(${literal(path)}, auto_type_candidates = ['BIGINT', 'DOUBLE', 'VARCHAR', 'DATE', 'TIMESTAMP'])`;
  await query(`CREATE OR REPLACE TABLE source AS SELECT * FROM ${reader}`);
}

export type ColumnInfo = {
  name: string;
  type: string;
  kind: "numerical" | "categorical" | "unsupported";
  distinct: number;
  nulls: number;
  integral: boolean;
};

const NUMERIC = /^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|UHUGEINT|FLOAT|DOUBLE|REAL|DECIMAL)/i;
const CATEGORICAL = /^(VARCHAR|BOOLEAN|ENUM)/i;

/** Columns of `view` with sdm's stype rules: numbers numerical, strings and booleans categorical. */
export async function describe(view: string): Promise<{ rows: number; columns: ColumnInfo[] }> {
  const schema = (await query(`DESCRIBE ${view}`)).toArray().map((r) => r.toJSON());
  const rows = Number((await query(`SELECT count(*) AS n FROM ${view}`)).get(0)!.toJSON().n);
  const columns: ColumnInfo[] = [];
  for (const s of schema) {
    const name = String(s.column_name);
    const type = String(s.column_type);
    const kind = NUMERIC.test(type) ? "numerical" : CATEGORICAL.test(type) ? "categorical" : "unsupported";
    const stats = (
      await query(
        `SELECT count(DISTINCT ${ident(name)}) AS d, count(*) - count(${ident(name)}) AS m` +
          (kind === "numerical" ? `, bool_and(${ident(name)} = round(${ident(name)})) AS i` : "") +
          ` FROM ${view}`,
      )
    ).get(0)!.toJSON();
    columns.push({ name, type, kind, distinct: Number(stats.d), nulls: Number(stats.m), integral: Boolean(stats.i) });
  }
  return { rows, columns };
}

/** Numerical column as float64 with NaN for missing; categorical as strings with null. */
export function numericColumn(table: Table, name: string): Float64Array {
  const vector = table.getChild(name)!;
  const out = new Float64Array(table.numRows);
  for (let i = 0; i < table.numRows; i++) {
    const v = vector.get(i);
    out[i] = v === null || v === undefined ? NaN : Number(v);
  }
  return out;
}

export function stringColumn(table: Table, name: string): (string | null)[] {
  const vector = table.getChild(name)!;
  const out: (string | null)[] = new Array(table.numRows);
  for (let i = 0; i < table.numRows; i++) {
    const v = vector.get(i);
    out[i] = v === null || v === undefined ? null : String(v);
  }
  return out;
}

/** Store JS rows as a DuckDB table (via a registered JSON file). */
export async function tableFromRows(name: string, rows: Record<string, unknown>[]): Promise<void> {
  await connect();
  const path = `${name}_${Date.now()}.json`;
  await db!.registerFileText(path, JSON.stringify(rows));
  await query(`CREATE OR REPLACE TABLE ${ident(name)} AS SELECT * FROM read_json_auto(${literal(path)})`);
}

/** CSV bytes of a query result. */
export async function exportCsv(sql: string): Promise<Uint8Array> {
  await connect();
  const path = `export_${Date.now()}.csv`;
  await query(`COPY (${sql}) TO ${literal(path)} (HEADER, DELIMITER ',')`);
  return db!.copyFileToBuffer(path);
}
