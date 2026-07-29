-- MySQL metadata schema for Tark AI (Layers 4-5)
-- Run inside the tark_metadata database.

USE tark_metadata;

-- Schema context: every column of every table
CREATE TABLE IF NOT EXISTS schema_context (
  id INT AUTO_INCREMENT PRIMARY KEY,
  table_name VARCHAR(255),
  column_name VARCHAR(255),
  data_type VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relationship context: discovered table joins
CREATE TABLE IF NOT EXISTS relationship_context (
  id INT AUTO_INCREMENT PRIMARY KEY,
  parent_table VARCHAR(255),
  child_table VARCHAR(255),
  join_key VARCHAR(255),
  confidence DECIMAL(5,2),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Business context: business term -> actual column mapping
CREATE TABLE IF NOT EXISTS business_context (
  id INT AUTO_INCREMENT PRIMARY KEY,
  business_term VARCHAR(255),
  table_name VARCHAR(255),
  column_name VARCHAR(255),
  aggregation VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- KPI context: named KPIs and how to compute them
CREATE TABLE IF NOT EXISTS kpi_context (
  id INT AUTO_INCREMENT PRIMARY KEY,
  kpi_name VARCHAR(255),
  table_name VARCHAR(255),
  expression VARCHAR(500),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query audit log (Layer 9 / Layer 15)
CREATE TABLE IF NOT EXISTS query_logs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_query TEXT,
  generated_sql TEXT,
  engine VARCHAR(50),
  intent VARCHAR(50),
  execution_time_sec DECIMAL(10,4),
  row_count INT,
  status VARCHAR(20),
  error_message TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
