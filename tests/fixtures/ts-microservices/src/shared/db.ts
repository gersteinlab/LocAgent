export interface QueryResult {
  rows: unknown[];
}

export class DatabaseClient {
  query(sql: string, params: unknown[] = []): QueryResult {
    return { rows: [] };
  }
}

export const db = new DatabaseClient();
