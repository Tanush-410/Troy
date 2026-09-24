"""Permission-log drift detector (PLDD).

Reads only DetectorEvent (policy/log_schema.py). Must not import labels, taint
fields, the harm oracle, the environment, or transcripts; a test enforces it.
"""
