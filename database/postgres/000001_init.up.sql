SET client_min_messages TO WARNING;
-- SET ROLE=:XXXX_admin;
SET search_path TO public;

-- Create function to automate update of `dt_modified` columns
CREATE FUNCTION set_dt_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.dt_modified = NOW();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

CREATE FUNCTION update_dt_modified_column(tablename REGCLASS)
RETURNS VOID AS $$
BEGIN
    EXECUTE FORMAT('CREATE TRIGGER set_dt_modified_column BEFORE UPDATE ON %s FOR EACH ROW WHEN (OLD IS DISTINCT FROM NEW) EXECUTE FUNCTION set_dt_modified_column();', CONCAT('"', tablename, '"'));
END;
$$ LANGUAGE 'plpgsql';

-- Create table to store chat user records
CREATE TABLE chat_user (
      user_id               SMALLINT            NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created            TIMESTAMPTZ         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified           TIMESTAMPTZ         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , email                 TEXT                NOT NULL
    , full_name             TEXT                NOT NULL
    , password_hash         TEXT                NOT NULL
    , PRIMARY KEY (user_id)
)
;
CREATE UNIQUE INDEX idx_chat_user_email         ON chat_user    USING btree (email);
SELECT update_dt_modified_column('chat_user');
COMMENT ON TABLE chat_user IS 'This table stores chat user records';
