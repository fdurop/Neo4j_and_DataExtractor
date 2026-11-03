# view_db.py
import sqlite3

DB_PATH = "users.db"  # 手动指定数据库路径

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print(f"📘 数据库路径: {DB_PATH}")
print("=" * 80)

# 获取所有表名
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

if not tables:
    print("⚠️ 数据库中没有任何表。")
else:
    for (table_name,) in tables:
        print(f"\n📂 表名: {table_name}")
        print("-" * 40)
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")

        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()
        print(f"\n📊 共 {len(rows)} 条记录:")
        for row in rows:
            print(row)

conn.close()
