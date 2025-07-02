CREATE DATABASE optimized_ai_system;

CREATE USER ai_user WITH PASSWORD 'ai_password';

GRANT ALL PRIVILEGES ON DATABASE optimized_ai_system TO ai_user;

-- Optional: Create tables for MLflow if not using auto-create
-- CREATE TABLE IF NOT EXISTS alembic_version (
--     version_num VARCHAR(32) NOT NULL PRIMARY KEY
-- );

-- CREATE TABLE IF NOT EXISTS experiments (
--     experiment_id SERIAL PRIMARY KEY,
--     name VARCHAR(256) NOT NULL,
--     artifact_location VARCHAR(256),
--     lifecycle_stage VARCHAR(32),
--     creation_time BIGINT,
--     last_update_time BIGINT
-- );

-- CREATE TABLE IF NOT EXISTS runs (
--     run_uuid VARCHAR(32) NOT NULL PRIMARY KEY,
--     experiment_id INTEGER NOT NULL,
--     name VARCHAR(256),
--     source_type VARCHAR(20),
--     source_name VARCHAR(500),
--     entry_point_name VARCHAR(50),
--     user_id VARCHAR(256),
--     status VARCHAR(20),
--     start_time BIGINT,
--     end_time BIGINT,
--     source_version VARCHAR(50),
--     lifecycle_stage VARCHAR(32),
--     artifact_uri VARCHAR(256),
--     run_id VARCHAR(32)
-- );

-- CREATE TABLE IF NOT EXISTS metrics (
--     key VARCHAR(250) NOT NULL,
--     value REAL NOT NULL,
--     timestamp BIGINT NOT NULL,
--     run_uuid VARCHAR(32) NOT NULL,
--     step BIGINT DEFAULT 0 NOT NULL,
--     is_nan BOOLEAN DEFAULT FALSE NOT NULL,
--     PRIMARY KEY (key, timestamp, step, run_uuid)
-- );

-- CREATE TABLE IF NOT EXISTS params (
--     key VARCHAR(250) NOT NULL,
--     value VARCHAR(250) NOT NULL,
--     run_uuid VARCHAR(32) NOT NULL,
--     PRIMARY KEY (key, run_uuid)
-- );

-- CREATE TABLE IF NOT EXISTS tags (
--     key VARCHAR(250) NOT NULL,
--     value VARCHAR(250),
--     run_uuid VARCHAR(32) NOT NULL,
--     PRIMARY KEY (key, run_uuid)
-- );

