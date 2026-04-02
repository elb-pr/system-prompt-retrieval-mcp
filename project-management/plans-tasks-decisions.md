# Plans, Tasks & Decisions

## How This System Works

This is a lightweight project management layer. It should live in `.claude/docs/` so it stays visible to Claude sessions without polluting the main project structure. Each new Plan must be titled with a name, date and label, any label you feel like. The tasks file or that plan must correspond to said label. The decisions file will then correspond to the same label. Creating a dependency chain of plans tasks and decisions that cannot be confused.

## Structure

```text
.claude/docs/
├── plans-tasks-decisions.md
├── sync.md
├── claude-personal-notes.md
├── plans/
│   ├── plan-date-label-template.md
│   └── plan-YYYY-MM-DD-label.md
├── tasks/
│   └── tasks.md
├── decisions/
│   └── decisions.md
└── session-logs/
    └── .gitignore
```

## Rules

1. Check tasks.md at the start of repo work
2. Update tasks.md when work changes
3. Put durable decisions in decisions.md, not chat
4. Create plans for multi-session or complex work
5. Use tasks.md directly for one-off work
