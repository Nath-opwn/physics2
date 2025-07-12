#!/usr/bin/env python3
"""
清空知识库表的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.database import SessionLocal
from src.models.models import KnowledgeItem

def clear_knowledge_table():
    """清空知识库表"""
    db = SessionLocal()
    try:
        count = db.query(KnowledgeItem).delete()
        db.commit()
        print(f"已清空知识库表，删除了 {count} 条记录。")
    except Exception as e:
        db.rollback()
        print(f"清空知识库表失败: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # 确认操作
    confirm = input("此操作将清空所有知识库内容，是否继续？(y/n): ")
    if confirm.lower() == 'y':
        clear_knowledge_table()
    else:
        print("操作已取消。") 