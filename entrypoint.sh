#!/bin/sh

# Wait for PostgreSQL to be available
if [ "$DATABASE" = "postgres" ]; then
    echo "Waiting for PostgreSQL..."

    while ! nc -z $DATABASE_HOST $DATABASE_PORT; do
      sleep 0.1
    done

    echo "PostgreSQL started"
fi

# Create the database and seed if in development environment
if [ "$LCNC_ENV" = "development" ]; then
    echo "Creating the database tables..."

    psql -h $DATABASE_HOST -U $DATABASE_USER -d $DATABASE_NAME <<EOF
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        session_id VARCHAR(255)
    );
EOF

    echo "Tables created"
fi

# Always create and seed the database
echo "Seeding the database..."

psql -h $DATABASE_HOST -U $DATABASE_USER -d $DATABASE_NAME <<EOF
    -- Example seed data
    INSERT INTO users (username, password) VALUES
    ('admin', 'password'),
    ('user', 'password')
    ON CONFLICT (username) DO NOTHING;
EOF

echo "Database seeded"
