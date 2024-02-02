import json

import psycopg2
from tabulate import tabulate


class PostgresDB:
    """
    A class to manage postgres connections and queries
    """

    def __init__(self):
        self.conn = None
        self.cur = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def connect_with_url(self, url):
        self.conn = psycopg2.connect(url)
        self.cur = self.conn.cursor()

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def run_sql(self, sql) -> str:
        """
        Run a SQL query against the postgres database.
        Returns JSON.
        """
        self.cur.execute(sql)
        columns = [desc[0] for desc in self.cur.description]
        res = self.cur.fetchall()

        list_of_dicts = [dict(zip(columns, row)) for row in res]

        json_result = json.dumps(list_of_dicts, indent=4)

        return json_result

    # method to run a sql and return markdown
    def run_sql_to_markdown(self, sql) -> str:
        """
        Run a SQL query against the postgres database
        Returns markdown table.
        """
        self.cur.execute(sql)
        columns = [desc[0] for desc in self.cur.description]
        res = self.cur.fetchall()

        list_of_dicts = [dict(zip(columns, row)) for row in res]

        markdown_table = self.to_markdown(list_of_dicts)
        print(markdown_table)
        return markdown_table

    @staticmethod
    def to_markdown(data) -> str:
        """
        Convert a list of dictionaries to markdown
        """
        return tabulate(data, headers="keys", tablefmt="pipe")
