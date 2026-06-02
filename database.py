import sqlite3
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "data/financial_system.db"

def get_db_connection():
    # Ensure raw directory exists (data/ processed and raw are there, so data/ exists)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );
    """)
    
    # Create user_state table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_state (
        user_id INTEGER UNIQUE NOT NULL,
        state TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)
    
    conn.commit()
    conn.close()
    print("SQLite Database initialized successfully.")

def register_user(email, password):
    email = email.strip().lower()
    hashed_pwd = generate_password_hash(password)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pwd))
        user_id = cursor.lastrowid
        
        # Initialize default empty state JSON
        default_state = json.dumps({
            "profile": None,
            "goals": [],
            "cards": [],
            "activities": []
        })
        cursor.execute("INSERT INTO user_state (user_id, state) VALUES (?, ?)", (user_id, default_state))
        
        conn.commit()
        return {"success": True, "user_id": user_id}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Email already exists"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()

def authenticate_user(email, password):
    email = email.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, email, password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user["password"], password):
        return {"id": user["id"], "email": user["email"]}
    return None

def save_user_state(user_id, state_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        state_str = json.dumps(state_dict)
        cursor.execute("""
        INSERT OR REPLACE INTO user_state (user_id, state) 
        VALUES (?, ?)
        """, (user_id, state_str))
        conn.commit()
        return True
    except Exception as e:
        print("Error saving state:", e)
        return False
    finally:
        conn.close()

def get_user_state(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT state FROM user_state WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        try:
            return json.loads(row["state"])
        except Exception:
            return None
    return None
