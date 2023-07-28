@echo off
@echo Running analyzer
python analyze_all_papers.py
@echo generating draft reports
python generate_research_report.py
@echo finalizing reports
python finalize_research_report.py
@echo evaluating work
python evaluate_work.py
@echo done