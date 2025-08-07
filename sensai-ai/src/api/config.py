import os
from os.path import exists
from api.models import LeaderboardViewType, TaskInputType, TaskAIResponseType, TaskType

if exists("/appdata"):
    data_root_dir = "/appdata"
    root_dir = "/demo"
    log_dir = "/appdata/logs"
else:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(root_dir)

    data_root_dir = f"{parent_dir}/db"
    log_dir = f"{parent_dir}/logs"

if not exists(data_root_dir):
    os.makedirs(data_root_dir)

if not exists(log_dir):
    os.makedirs(log_dir)


sqlite_db_path = f"{data_root_dir}/db.sqlite"
log_file_path = f"{log_dir}/backend.log"

chat_history_table_name = "chat_history"
tasks_table_name = "tasks"
questions_table_name = "questions"
blocks_table_name = "blocks"
tests_table_name = "tests"
cohorts_table_name = "cohorts"
course_tasks_table_name = "course_tasks"
course_milestones_table_name = "course_milestones"
courses_table_name = "courses"
course_cohorts_table_name = "course_cohorts"
task_scoring_criteria_table_name = "task_scoring_criteria"
groups_table_name = "groups"
user_cohorts_table_name = "user_cohorts"
user_groups_table_name = "user_groups"
milestones_table_name = "milestones"
tags_table_name = "tags"
task_tags_table_name = "task_tags"
users_table_name = "users"
badges_table_name = "badges"
cv_review_usage_table_name = "cv_review_usage"
organizations_table_name = "organizations"
user_organizations_table_name = "user_organizations"
task_completions_table_name = "task_completions"
scorecards_table_name = "scorecards"
question_scorecards_table_name = "question_scorecards"
group_role_learner = "learner"
group_role_mentor = "mentor"
course_generation_jobs_table_name = "course_generation_jobs"
task_generation_jobs_table_name = "task_generation_jobs"
org_api_keys_table_name = "org_api_keys"
code_drafts_table_name = "code_drafts"

UPLOAD_FOLDER_NAME = "uploads"

uncategorized_milestone_name = "[UNASSIGNED]"
uncategorized_milestone_color = "#808080"

# Flag to use Google services instead of OpenAI
# When False: Uses OpenAI Whisper API for audio transcription + gpt-4o-mini for chat
# When True: Uses Google Speech-to-Text + Google Gemini for chat (requires Google credentials)
USE_GOOGLE_SERVICES = os.getenv("USE_GOOGLE_SERVICES", "false").lower() == "true"

openai_plan_to_model_name = {
    "reasoning": "openai/gpt-4o-mini",
    "text": "openai/gpt-4o-mini",
    "text-mini": "openai/gpt-4o-mini",
    "audio": "openai/gpt-4o-mini",  # Use gpt-4o-mini since API key doesn't have access to gpt-4o
    "router": "openai/gpt-4o-mini",
}

# Google Gemini model configuration
google_plan_to_model_name = {
    "reasoning": "gemini-1.5-pro",  # For complex reasoning tasks
    "text": "gemini-1.5-pro",       # For general text tasks
    "text-mini": "gemini-1.5-flash", # For faster, simpler tasks
    "audio": "gemini-1.5-pro",      # For audio transcription and analysis
    "router": "gemini-1.5-flash",   # For routing decisions
}
