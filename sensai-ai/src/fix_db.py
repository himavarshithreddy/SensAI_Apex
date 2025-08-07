import asyncio
import os
from api.db import (
    create_organizations_table,
    create_org_api_keys_table,
    create_users_table,
    create_user_organizations_table,
    create_milestones_table,
    create_cohort_tables,
    create_courses_table,
    create_course_cohorts_table,
    create_tasks_table,
    create_questions_table,
    create_scorecards_table,
    create_question_scorecards_table,
    create_chat_history_table,
    create_task_completion_table,
    create_course_tasks_table,
    create_course_milestones_table,
    create_course_generation_jobs_table,
    create_task_generation_jobs_table,
    create_code_drafts_table,
)
from api.utils.db import get_new_db_connection, check_table_exists
from api.config import (
    organizations_table_name,
    org_api_keys_table_name,
    users_table_name,
    user_organizations_table_name,
    milestones_table_name,
    cohorts_table_name,
    user_cohorts_table_name,
    courses_table_name,
    course_cohorts_table_name,
    tasks_table_name,
    questions_table_name,
    scorecards_table_name,
    question_scorecards_table_name,
    chat_history_table_name,
    task_completions_table_name,
    course_tasks_table_name,
    course_milestones_table_name,
    course_generation_jobs_table_name,
    task_generation_jobs_table_name,
    code_drafts_table_name,
)


async def fix_database():
    """Create all missing tables in the database"""
    print("Starting database fix...")
    
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        
        # List of all tables that should exist
        tables_to_create = [
            (organizations_table_name, create_organizations_table),
            (org_api_keys_table_name, create_org_api_keys_table),
            (users_table_name, create_users_table),
            (user_organizations_table_name, create_user_organizations_table),
            (milestones_table_name, create_milestones_table),
            (cohorts_table_name, create_cohort_tables),  # This creates both cohorts and user_cohorts
            (courses_table_name, create_courses_table),
            (course_cohorts_table_name, create_course_cohorts_table),
            (tasks_table_name, create_tasks_table),
            (questions_table_name, create_questions_table),
            (scorecards_table_name, create_scorecards_table),
            (question_scorecards_table_name, create_question_scorecards_table),
            (chat_history_table_name, create_chat_history_table),
            (task_completions_table_name, create_task_completion_table),
            (course_tasks_table_name, create_course_tasks_table),
            (course_milestones_table_name, create_course_milestones_table),
            (course_generation_jobs_table_name, create_course_generation_jobs_table),
            (task_generation_jobs_table_name, create_task_generation_jobs_table),
            (code_drafts_table_name, create_code_drafts_table),
        ]
        
        # Check and create missing tables
        for table_name, create_func in tables_to_create:
            if not await check_table_exists(table_name, cursor):
                print(f"Creating table: {table_name}")
                await create_func(cursor)
            else:
                print(f"Table already exists: {table_name}")
        
        await conn.commit()
        print("Database fix completed!")


if __name__ == "__main__":
    asyncio.run(fix_database()) 