-- ===========================
-- 1. CREATE TABLES
-- ===========================

CREATE TABLE DRIVER (
    driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    data_type TEXT,
    description TEXT
);

CREATE TABLE RULE (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT
);

CREATE TABLE RULE_VERSION (
    rule_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL,
    version_no INTEGER NOT NULL,
    status TEXT,               -- DRAFT, ACTIVE, RETIRED
    valid_from TEXT,           -- ISO date string
    valid_to TEXT,             -- ISO date string
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT,
    FOREIGN KEY (rule_id) REFERENCES RULE(rule_id)
);

CREATE TABLE RULE_INPUT (
    rule_input_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_version_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    value_type TEXT,           -- SINGLE, GROUP, SET
    created_by TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (rule_version_id) REFERENCES RULE_VERSION(rule_version_id),
    FOREIGN KEY (driver_id) REFERENCES DRIVER(driver_id)
);

CREATE TABLE RULE_INPUT_VALUE (
    input_value_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_input_id INTEGER NOT NULL,
    value TEXT,
    group_id INTEGER,
    created_by TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (rule_input_id) REFERENCES RULE_INPUT(rule_input_id)
);

CREATE TABLE RULE_OUTPUT (
    rule_output_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_version_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    value TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (rule_version_id) REFERENCES RULE_VERSION(rule_version_id),
    FOREIGN KEY (driver_id) REFERENCES DRIVER(driver_id)
);

CREATE TABLE RULESET (
    ruleset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE RULESET_VERSION (
    ruleset_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ruleset_id INTEGER NOT NULL,
    version_no INTEGER NOT NULL,
    status TEXT,               -- DRAFT, ACTIVE, INACTIVE
    valid_from TEXT,
    valid_to TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT,
    FOREIGN KEY (ruleset_id) REFERENCES RULESET(ruleset_id)
);

CREATE TABLE RULESET_RULE (
    ruleset_version_id INTEGER NOT NULL,
    rule_version_id INTEGER NOT NULL,
    PRIMARY KEY (ruleset_version_id, rule_version_id),
    FOREIGN KEY (ruleset_version_id) REFERENCES RULESET_VERSION(ruleset_version_id),
    FOREIGN KEY (rule_version_id) REFERENCES RULE_VERSION(rule_version_id)
);

-- ===========================
-- 2. INSERT SAMPLE DATA
-- ===========================

-- Drivers
INSERT INTO DRIVER (name, data_type, description) VALUES
('STUDENT_NAME', 'TEXT', 'Name of student'),
('BRANCH', 'TEXT', 'Branch of study'),
('CITY', 'TEXT', 'City of student'),
('CLASS_TEACHER', 'TEXT', 'Assigned class teacher');

-- Rule
INSERT INTO RULE (description) VALUES
('Assign class teacher based on branch and city');

-- Rule Versions
-- R1 v1 (Delhi)
INSERT INTO RULE_VERSION 
(rule_id, version_no, status, valid_from, created_by, created_at) 
VALUES 
(1, 1, 'ACTIVE', '2025-01-01', 'admin', '2025-01-01');

-- R1 v2 (Mumbai)
INSERT INTO RULE_VERSION 
(rule_id, version_no, status, valid_from, created_by, created_at) 
VALUES 
(1, 2, 'ACTIVE', '2025-12-17', 'admin', '2025-12-17');

-- Rule Inputs for v1
INSERT INTO RULE_INPUT (rule_version_id, driver_id, value_type, created_by, created_at) VALUES
(1, 1, 'SINGLE', 'admin', '2025-01-01'),
(1, 2, 'GROUP', 'admin', '2025-01-01'),
(1, 3, 'SINGLE', 'admin', '2025-01-01');

-- Rule Inputs for v2
INSERT INTO RULE_INPUT (rule_version_id, driver_id, value_type, created_by, created_at) VALUES
(2, 1, 'SINGLE', 'admin', '2025-12-17'),
(2, 2, 'GROUP', 'admin', '2025-12-17'),
(2, 3, 'SINGLE', 'admin', '2025-12-17');

-- Rule Input Values v1
INSERT INTO RULE_INPUT_VALUE (rule_input_id, value, group_id, created_by, created_at) VALUES
(1, '*', NULL, 'admin', '2025-01-01'),
(2, 'Maths', 1, 'admin', '2025-01-01'),
(2, 'Science', 1, 'admin', '2025-01-01'),
(3, 'Delhi', NULL, 'admin', '2025-01-01');

-- Rule Input Values v2
INSERT INTO RULE_INPUT_VALUE (rule_input_id, value, group_id, created_by, created_at) VALUES
(4, '*', NULL, 'admin', '2025-12-17'),
(5, 'Maths', 2, 'admin', '2025-12-17'),
(5, 'Science', 2, 'admin', '2025-12-17'),
(6, 'Mumbai', NULL, 'admin', '2025-12-17');

-- Rule Outputs
INSERT INTO RULE_OUTPUT (rule_version_id, driver_id, value, created_by, created_at) VALUES
(1, 4, 'Mr. Sharma', 'admin', '2025-01-01'),
(2, 4, 'Mr. Sharma', 'admin', '2025-12-17');

-- Ruleset
INSERT INTO RULESET (name, description) VALUES
('Student Teacher Assignment', 'Assign teachers based on branch and city');

-- Ruleset Versions
INSERT INTO RULESET_VERSION (ruleset_id, version_no, status, valid_from, created_by, created_at) VALUES
(1, 1, 'ACTIVE', '2025-01-01', 'admin', '2025-01-01'),
(1, 2, 'ACTIVE', '2025-12-17', 'admin', '2025-12-17'),
(1, 3, 'ACTIVE', '2025-12-21', 'admin', '2025-12-21');

-- Ruleset Rules
INSERT INTO RULESET_RULE (ruleset_version_id, rule_version_id) VALUES
(1, 1),
(2, 2),
(3, 1); -- rollback

-- ===========================
-- 3. SAMPLE SELECT QUERY
-- Get Class Teacher for student with branch and city using latest active ruleset
-- ===========================

WITH active_ruleset AS (
    SELECT ruleset_version_id 
    FROM RULESET_VERSION 
    WHERE ruleset_id = 1 AND status = 'ACTIVE'
    ORDER BY valid_from DESC
    LIMIT 1
),
active_rules AS (
    SELECT rr.rule_version_id
    FROM RULESET_RULE rr
    JOIN active_ruleset ar ON rr.ruleset_version_id = ar.ruleset_version_id
)
SELECT ro.value AS class_teacher
FROM RULE_OUTPUT ro
JOIN active_rules ar ON ro.rule_version_id = ar.rule_version_id
WHERE ro.driver_id = 4;  -- CLASS_TEACHER





-- ===========================
-- 1. INSERT ROLLBACK RULESET VERSION
-- ===========================

-- Create a new ruleset version that points back to R1 v1
INSERT INTO RULESET_VERSION 
(ruleset_id, version_no, status, valid_from, created_by, created_at)
VALUES 
(1, 4, 'ACTIVE', '2025-12-25', 'admin', '2025-12-25');

-- Link it to R1 v1
INSERT INTO RULESET_RULE (ruleset_version_id, rule_version_id)
VALUES (4, 1);

-- ===========================
-- 2. SELECT CLASS TEACHER USING LATEST ACTIVE RULESET (WITH ROLLBACK)
-- ===========================

WITH active_ruleset AS (
    SELECT ruleset_version_id 
    FROM RULESET_VERSION 
    WHERE ruleset_id = 1 AND status = 'ACTIVE'
    ORDER BY valid_from DESC
    LIMIT 1
),
active_rules AS (
    SELECT rr.rule_version_id
    FROM RULESET_RULE rr
    JOIN active_ruleset ar ON rr.ruleset_version_id = ar.ruleset_version_id
)
SELECT ro.value AS class_teacher
FROM RULE_OUTPUT ro
JOIN active_rules ar ON ro.rule_version_id = ar.rule_version_id
WHERE ro.driver_id = 4;  -- CLASS_TEACHER


-- ===========================
-- 1. CREATE TABLES
-- ===========================

CREATE TABLE DRIVER (
    driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    data_type TEXT,
    description TEXT
);

CREATE TABLE RULE (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT
);

CREATE TABLE RULE_VERSION (
    rule_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL,
    version_no INTEGER NOT NULL,
    status TEXT,               -- DRAFT, ACTIVE, RETIRED
    valid_from TEXT,           -- ISO date string
    valid_to TEXT,             -- ISO date string
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT,
    FOREIGN KEY (rule_id) REFERENCES RULE(rule_id)
);

CREATE TABLE RULE_INPUT (
    rule_input_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_version_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    value_type TEXT,           -- SINGLE, GROUP, SET
    created_by TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (rule_version_id) REFERENCES RULE_VERSION(rule_version_id),
    FOREIGN KEY (driver_id) REFERENCES DRIVER(driver_id)
);

CREATE TABLE RULE_INPUT_VALUE (
    input_value_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_input_id INTEGER NOT NULL,
    value TEXT,
    group_id INTEGER,
    created_by TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (rule_input_id) REFERENCES RULE_INPUT(rule_input_id)
);

CREATE TABLE RULE_OUTPUT (
    rule_output_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_version_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    value TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (rule_version_id) REFERENCES RULE_VERSION(rule_version_id),
    FOREIGN KEY (driver_id) REFERENCES DRIVER(driver_id)
);

CREATE TABLE RULESET (
    ruleset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE RULESET_VERSION (
    ruleset_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ruleset_id INTEGER NOT NULL,
    version_no INTEGER NOT NULL,
    status TEXT,               -- DRAFT, ACTIVE, INACTIVE
    valid_from TEXT,
    valid_to TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT,
    FOREIGN KEY (ruleset_id) REFERENCES RULESET(ruleset_id)
);

CREATE TABLE RULESET_RULE (
    ruleset_version_id INTEGER NOT NULL,
    rule_version_id INTEGER NOT NULL,
    PRIMARY KEY (ruleset_version_id, rule_version_id),
    FOREIGN KEY (ruleset_version_id) REFERENCES RULESET_VERSION(ruleset_version_id),
    FOREIGN KEY (rule_version_id) REFERENCES RULE_VERSION(rule_version_id)
);

-- ===========================
-- 2. INSERT SAMPLE DATA
-- ===========================

-- Drivers
INSERT INTO DRIVER (name, data_type, description) VALUES
('STUDENT_NAME', 'TEXT', 'Name of student'),
('BRANCH', 'TEXT', 'Branch of study'),
('CITY', 'TEXT', 'City of student'),
('CLASS_TEACHER', 'TEXT', 'Assigned class teacher');

-- Rule
INSERT INTO RULE (description) VALUES
('Assign class teacher based on branch and city');

-- Rule Versions
-- R1 v1 (Delhi)
INSERT INTO RULE_VERSION 
(rule_id, version_no, status, valid_from, created_by, created_at) 
VALUES 
(1, 1, 'ACTIVE', '2025-01-01', 'admin', '2025-01-01');

-- R1 v2 (Mumbai)
INSERT INTO RULE_VERSION 
(rule_id, version_no, status, valid_from, created_by, created_at) 
VALUES 
(1, 2, 'ACTIVE', '2025-12-17', 'admin', '2025-12-17');

-- Rule Inputs for v1
INSERT INTO RULE_INPUT (rule_version_id, driver_id, value_type, created_by, created_at) VALUES
(1, 1, 'SINGLE', 'admin', '2025-01-01'),
(1, 2, 'GROUP', 'admin', '2025-01-01'),
(1, 3, 'SINGLE', 'admin', '2025-01-01');

-- Rule Inputs for v2
INSERT INTO RULE_INPUT (rule_version_id, driver_id, value_type, created_by, created_at) VALUES
(2, 1, 'SINGLE', 'admin', '2025-12-17'),
(2, 2, 'GROUP', 'admin', '2025-12-17'),
(2, 3, 'SINGLE', 'admin', '2025-12-17');

-- Rule Input Values v1
INSERT INTO RULE_INPUT_VALUE (rule_input_id, value, group_id, created_by, created_at) VALUES
(1, '*', NULL, 'admin', '2025-01-01'),
(2, 'Maths', 1, 'admin', '2025-01-01'),
(2, 'Science', 1, 'admin', '2025-01-01'),
(3, 'Delhi', NULL, 'admin', '2025-01-01');

-- Rule Input Values v2
INSERT INTO RULE_INPUT_VALUE (rule_input_id, value, group_id, created_by, created_at) VALUES
(4, '*', NULL, 'admin', '2025-12-17'),
(5, 'Maths', 2, 'admin', '2025-12-17'),
(5, 'Science', 2, 'admin', '2025-12-17'),
(6, 'Mumbai', NULL, 'admin', '2025-12-17');

-- Rule Outputs
INSERT INTO RULE_OUTPUT (rule_version_id, driver_id, value, created_by, created_at) VALUES
(1, 4, 'Mr. Sharma', 'admin', '2025-01-01'),
(2, 4, 'Mr. Sharma', 'admin', '2025-12-17');

-- Ruleset
INSERT INTO RULESET (name, description) VALUES
('Student Teacher Assignment', 'Assign teachers based on branch and city');

-- Ruleset Versions
INSERT INTO RULESET_VERSION (ruleset_id, version_no, status, valid_from, created_by, created_at) VALUES
(1, 1, 'ACTIVE', '2025-01-01', 'admin', '2025-01-01'),
(1, 2, 'ACTIVE', '2025-12-17', 'admin', '2025-12-17'),
(1, 3, 'ACTIVE', '2025-12-21', 'admin', '2025-12-21');

-- Ruleset Rules
INSERT INTO RULESET_RULE (ruleset_version_id, rule_version_id) VALUES
(1, 1),
(2, 2),
(3, 1); -- rollback

-- ===========================
-- 3. SAMPLE SELECT QUERY
-- Get Class Teacher for student with branch and city using latest active ruleset
-- ===========================

WITH active_ruleset AS (
    SELECT ruleset_version_id 
    FROM RULESET_VERSION 
    WHERE ruleset_id = 1 AND status = 'ACTIVE'
    ORDER BY valid_from DESC
    LIMIT 1
),
active_rules AS (
    SELECT rr.rule_version_id
    FROM RULESET_RULE rr
    JOIN active_ruleset ar ON rr.ruleset_version_id = ar.ruleset_version_id
)
SELECT ro.value AS class_teacher
FROM RULE_OUTPUT ro
JOIN active_rules ar ON ro.rule_version_id = ar.rule_version_id
WHERE ro.driver_id = 4;  -- CLASS_TEACHER





-- ===========================
-- 1. INSERT ROLLBACK RULESET VERSION
-- ===========================

-- Create a new ruleset version that points back to R1 v1
INSERT INTO RULESET_VERSION 
(ruleset_id, version_no, status, valid_from, created_by, created_at)
VALUES 
(1, 4, 'ACTIVE', '2025-12-25', 'admin', '2025-12-25');

-- Link it to R1 v1
INSERT INTO RULESET_RULE (ruleset_version_id, rule_version_id)
VALUES (4, 1);

-- ===========================
-- 2. SELECT CLASS TEACHER USING LATEST ACTIVE RULESET (WITH ROLLBACK)
-- ===========================

WITH active_ruleset AS (
    SELECT ruleset_version_id 
    FROM RULESET_VERSION 
    WHERE ruleset_id = 1 AND status = 'ACTIVE'
    ORDER BY valid_from DESC
    LIMIT 1
),
active_rules AS (
    SELECT rr.rule_version_id
    FROM RULESET_RULE rr
    JOIN active_ruleset ar ON rr.ruleset_version_id = ar.ruleset_version_id
)
SELECT ro.value AS class_teacher
FROM RULE_OUTPUT ro
JOIN active_rules ar ON ro.rule_version_id = ar.rule_version_id
WHERE ro.driver_id = 4;  -- CLASS_TEACHER



-- ===========================
-- SELECT ACTIVE RULES WITH INPUT TYPE AND CLEAN TABLE FORMAT
-- ===========================

WITH latest_active_ruleset AS (
    SELECT ruleset_version_id
    FROM RULESET_VERSION
    WHERE status = 'ACTIVE'
    ORDER BY valid_from DESC
    LIMIT 1
),
active_rules AS (
    SELECT rr.rule_version_id, rv.version_no, r.description AS rule_desc
    FROM RULESET_RULE rr
    JOIN latest_active_ruleset lr ON rr.ruleset_version_id = lr.ruleset_version_id
    JOIN RULE_VERSION rv ON rr.rule_version_id = rv.rule_version_id
    JOIN RULE r ON rv.rule_id = r.rule_id
    WHERE rv.status = 'ACTIVE'
),
rule_inputs AS (
    SELECT ri.rule_version_id, d.name AS driver_name, 
           ri.value_type, riv.value, riv.group_id
    FROM RULE_INPUT ri
    JOIN RULE_INPUT_VALUE riv ON ri.rule_input_id = riv.rule_input_id
    JOIN DRIVER d ON ri.driver_id = d.driver_id
),
rule_outputs AS (
    SELECT ro.rule_version_id, d.name AS driver_name, ro.value
    FROM RULE_OUTPUT ro
    JOIN DRIVER d ON ro.driver_id = d.driver_id
)
SELECT 
    ar.rule_version_id,
    ar.version_no,
    ar.rule_desc,
    -- Group inputs per driver with type
    GROUP_CONCAT(ri.driver_name || ' (' || ri.value_type || '): ' || ri.value, ', ') AS inputs,
    GROUP_CONCAT(ro.driver_name || ': ' || ro.value, ', ') AS outputs
FROM active_rules ar
LEFT JOIN rule_inputs ri ON ar.rule_version_id = ri.rule_version_id
LEFT JOIN rule_outputs ro ON ar.rule_version_id = ro.rule_version_id
GROUP BY ar.rule_version_id, ar.version_no, ar.rule_desc
ORDER BY ar.rule_version_id;




