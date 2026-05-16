# database

Database migrations applied at startup by `migrate/migrate` containers in `docker-compose`.

- `postgres/<NNNNNN>_<name>.up.sql` — applied automatically by the `postgres-migrate` service. Current schema: a `chat_user` table for web-harness auth.
- `neo4j/<NNNNNN>_<name>.up.cypher` — applied by the `neo4j-migrate` service. Defines uniqueness constraints on `NamedEntity` and `Topic` nodes for the topic similarity graph.

Filenames follow the `migrate/migrate` convention: `NNNNNN_<name>.up.<ext>`. Migrations are applied in order and are forward-only (no `.down` files); roll back by writing a new forward migration.
