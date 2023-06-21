# How to use the Data Access Layer

The `DataAccessLayer` object is a wrapper to interact with connections to databases. Under the hood, it uses SQLAlchemy.

There are mainly two types:

- `DataAccessLayer`: generic DAL for any database (snowflake, postgres, etc)
- `InMemoryDataAccessLayer`: DAL for in-memory sqlite databases. Useful for tests as it has the same interface as the `DataAccessLayer`.

## Basic usage

```python
import pandas as pd

from data_access_layer import DataAccessLayer

conn_string = "snowflake://user:pass@hostname/schema/table"
dal = DataAccessLayer().connect(conn_string)
query = "SELECT * FROM my_table LIMIT 1000"
df = dal.query(query)
```
