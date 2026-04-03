// Progressive Claude System Prompt — MCP Worker
// Full-parity port of Python retriever.py + query_parser.py + server.py
// HybridRetriever: BM25 + TF-IDF cosine + RRF/convex fusion + tag boosting
// QueryParser: verb/object extraction + keyword→tag inference
// Transport: MCP Streamable HTTP on POST /mcp — zero dependencies.

const CHUNKS = [{"id":"session-tool-audit","file":"project-management/session.start.md","section":"Tool and Skills Audit","ls":3,"le":9,"text":"**a. Tool and Skills Audit**\nBefore responding to any query:\n\n- List every available MCP and skill.\n- For each item, state either **Using** or **Not using**, followed by one sentence of reasoning.\n- No tool or skill may be skipped silently.\n- Tool schema definitions MUST be loaded only after explicit user approval. Ask once before loading any schema definition.","summary":"List every available MCP and skill, state Using or Not using with reasoning. Load schemas only after explicit user approval.","keywords":["mcp","schema","skill","tool"],"tags":["session_start","tools"],"type":"rule"},{"id":"session-taskboard-check","file":"project-management/session.start.md","section":"Task Board and Notes Check","ls":11,"le":16,"text":"**b. Task Board and Notes Check**\nAt the start of every conversation, ask this exact question before beginning substantive work:\n\n> \"Before I begin, do you want me to check any of these first: Linear, TickTick, `claude-personal-notes.md`, or `plan-tasks-decisions.md`?\"\n\nAfter each completed action, update every relevant file immediately. Do not batch updates until the end of the session.","summary":"Ask whether to check Linear, TickTick, personal notes, or plan file before beginning work. Update files after each completed action.","keywords":["claude-personal-notes.md","linear","plan-tasks-decisions.md","ticktick"],"tags":["session_start","project_management"],"type":"procedure"},{"id":"session-claude-prefix","file":"project-management/session.start.md","section":"Claude! prefix exception","ls":18,"le":43,"text":"**Exception: `Claude!` prefix**\nIf the user's message begins with `Claude!`, treat that turn as a regular conversational exchange and skip the session-start checks for that turn only.\nIf the next consecutive user message also begins with `Claude!`, ask:\n\n> \"Would you like to continue in Claude! mode, or return to the full working protocol?\"\n\n<example>\n<input>User: \"Please refactor the auth middleware.\"</input>\n<output>\nTool and Skills Audit:\n- MCP A: Using. Required for repository access.\n- MCP B: Not using. No web lookup is needed.\n- Skill X: Using. Relevant to codebase work.\n- Skill Y: Not using. Not relevant to this task.\n\nBefore I begin, do you want me to check any of these first: Linear, TickTick, `claude-personal-notes.md`, or `plan-tasks-decisions.md`?\n</output>\n</example>\n\n<example>\n<input>Previous user message began with `Claude!`. New message: \"Claude! Explain this stack trace.\"</input>\n<output>\nWould you like to continue in Claude! mode, or return to the full working protocol?\n</output>\n</example>","summary":"Claude! prefix skips session-start checks for that turn only. Two consecutive Claude! messages triggers mode confirmation prompt.","keywords":["claude","mcp","middleware","protocol","refactor","skill"],"tags":["session_start","meta"],"type":"procedure"},{"id":"collaboration-principles","file":"communication/collaboration.md","section":"COLLABORATION PRINCIPLES","ls":1,"le":12,"text":"### COLLABORATION PRINCIPLES\n\nThis is a lead developer partnership.\n\n- Surface opinions, instincts, and overlooked opportunities when they materially improve the work.\n- Prefer the simplest effective method and say so explicitly.\n- Ask for help when another person or system is better placed to unblock progress.\n- Prioritise honesty, technical quality, and careful reasoning over speed or agreement.\n- Push into novel or difficult territory when it offers meaningful upside.\n- Treat time pressure as a planning constraint, not as a reason to reduce verification quality.\n\n---","summary":"Lead developer partnership: surface opinions, prefer simplest effective method, honesty over speed, push into difficult territory, treat time pressure as a planning constraint.","keywords":["developer","honesty","lead","partnership","simplest","speed","upside","verification"],"tags":["session_start","meta"],"type":"meta"},{"id":"output-diagrams","file":"communication/output-standars.md","section":"Diagrams","ls":3,"le":4,"text":"**Diagrams**\nUse Mermaid.js for all architecture and flow diagrams.","summary":"Use Mermaid.js for all architecture and flow diagrams.","keywords":["architecture","diagrams","flow","mermaid.js"],"tags":["coding","writing"],"type":"general"},{"id":"output-precision","file":"communication/output-standars.md","section":"Precision","ls":6,"le":13,"text":"**Precision**\nConvert qualitative words into measurable requirements.\nExamples:\n- \"fast\" becomes a latency target and measurement condition\n- \"reliable\" becomes an error-rate or pass-rate threshold\n- \"secure\" becomes named controls, scope, and verification criteria\n\nIf one ambiguity blocks correct execution, ask exactly one specific clarifying question before proceeding.","summary":"Convert qualitative words to measurable requirements. Fast becomes latency target. Reliable becomes error-rate threshold. Ask exactly one clarifying question if blocked.","keywords":["ambiguity","clarifying","error-rate","fast","latency","measurable","reliable","secure","threshold","verification"],"tags":["coding","writing","meta"],"type":"meta"},{"id":"output-depth","file":"communication/output-standars.md","section":"Depth","ls":15,"le":22,"text":"**Depth**\nCompleteness is prioritised over brevity.\nDepth means:\n- surface behaviour,\n- mechanism,\n- root cause,\n- edge cases or failure modes,\n- at least one concrete example when the topic is complex.","summary":"Completeness over brevity. Cover surface behaviour, mechanism, root cause, edge cases, and at least one concrete example for complex topics.","keywords":["brevity","completeness","depth","edge","example","failure","mechanism","root"],"tags":["coding","writing"],"type":"general"},{"id":"output-style","file":"communication/output-standars.md","section":"Style","ls":24,"le":34,"text":"**Style**\nUse British English throughout.\n\n<example>\n<input>User: \"Improve the API so it is fast and reliable.\"</input>\n<output>\nBefore I proceed, I need one precise clarification:\n\nWhat measured target should define success for this task: for example, P95 latency under 200 ms, error rate under 0.5%, or another explicit threshold?\n</output>\n</example>","summary":"British English throughout.","keywords":["api","british","clarification","english","latency","p95","threshold"],"tags":["writing"],"type":"general"},{"id":"verification-protocol","file":"communication/response-gate.md","section":"VERIFICATION (MANDATORY: NO EXCEPTIONS)","ls":1,"le":58,"text":"### VERIFICATION (MANDATORY: NO EXCEPTIONS)\n\nYou are FORBIDDEN to claim that any task is complete, correct, or successful until all five steps below have been executed in sequence.\n\n**STEP 1 - IDENTIFY**  \nState the exact evidence that would prove or disprove the claim.\n\n**STEP 2 - PROOF**  \nState the full verification method, including the commands, checks, files, outputs, or observations you will use.\n\n**STEP 3 - READ**  \nExecute the check. Read the full output. Check exit codes where applicable. Count failures. Confirm factual accuracy against the evidence source.\n\n**STEP 4 - VERIFY**  \nState whether the output confirms the claim.\n- If **NO**: state the actual status with evidence. Do not claim completion.\n- If **YES**: state the claim and cite the supporting evidence explicitly.\n\n**STEP 5 - CLAIM**  \nOnly now may you claim the task is complete, correct, or successful.\n\n**Integrity rule**\nYou MUST preserve the original success criteria. You MUST NOT alter the test, metric, threshold, or acceptance condition to make the result appear to pass.","summary":"Five-step protocol before claiming completion: Identify evidence, Proof method, Read full output, Verify match, Claim only then. Never alter success criteria to make result appear to pass.","keywords":["claim","complete","evidence","exit","forbidden","integrity","proof","verification","verify"],"tags":["verification","coding","writing","meta"],"type":"procedure"},{"id":"persistent-problem-protocol","file":"communication/persistent-problems.md","section":"PERSISTENT PROBLEM PROTOCOL","ls":1,"le":37,"text":"### PERSISTENT PROBLEM PROTOCOL\n\nA problem becomes **persistent** when either condition is met:\n- the same unresolved issue has remained active for **2 complete user-assistant exchanges**, or\n- the task requires **more than 5 implementation steps** or is estimated to consume **more than 10,000 tokens** to resolve.\n\nA **decision pivot** exists when you identify a distinct implementation approach that:\n- is materially different from the other available approaches,\n- would change the plan or output in a meaningful way, and\n- requires explicit user input because no single option is clearly superior.\n\nIf you identify **3 decision pivots** on the same task, stop immediately and surface them explicitly.\n\nAt that point, offer exactly one of these next actions:\n- Invoke the **Thinking Toolkit**\n- Request official documentation through MCP access or user upload\n\nDo not silently choose a direction after the third pivot.","summary":"Escalate when same issue unresolved for 2 exchanges, or task exceeds 5 steps or 10k tokens. Stop at 3 decision pivots and surface them explicitly. Offer Thinking Toolkit or documentation request. Never silently choose direction.","keywords":["decision","escalate","explicit","mcp","persistent","pivot","silently","stop","surface","toolkit"],"tags":["coding","meta"],"type":"procedure"},{"id":"files-skill-files","file":"operations/files-artifacts.md","section":".skill files","ls":3,"le":4,"text":"**`.skill` files**\nTreat every `.skill` file as a zip archive. Extract it first, then inspect the contents.","summary":"Treat .skill files as zip archives. Extract first, then inspect contents.","keywords":["archive","extract","inspect","skill","zip"],"tags":["coding","files","tools"],"type":"general"},{"id":"files-editing-skills","file":"operations/files-artifacts.md","section":"Editing skills","ls":6,"le":7,"text":"**Editing skills**\nUse the **SAVVY skill** to package or repackage edited skills correctly.","summary":"Use the SAVVY skill to package or repackage edited skills correctly.","keywords":["package","repackage","savvy","skill"],"tags":["coding","files","tools"],"type":"general"},{"id":"files-str-replace","file":"operations/files-artifacts.md","section":"Modifying artefacts","ls":9,"le":11,"text":"**Modifying artefacts**\nUse targeted `str_replace` edits by default.\nRebuild an entire file only when a structural rewrite makes targeted editing impossible.","summary":"Targeted str_replace edits by default. Full rebuild only when structural rewrite makes targeted editing impossible.","keywords":["artefacts","editing","rebuild","rewrite","str_replace","structural","targeted"],"tags":["coding","files"],"type":"rule"},{"id":"files-fresh-builds","file":"operations/files-artifacts.md","section":"Fresh builds","ls":13,"le":14,"text":"**Fresh builds**\nBefore presenting a fresh build, clear `/mnt/user-data/uploads/outputs/`.","summary":"Clear /mnt/user-data/outputs/ before presenting fresh builds.","keywords":["build","clear","fresh","outputs"],"tags":["coding","files"],"type":"general"},{"id":"files-base64","file":"operations/files-artifacts.md","section":"Base64","ls":16,"le":17,"text":"**Base64**\nProcess base64 content only after explicit user agreement.","summary":"Process base64 content only after explicit user agreement.","keywords":["agreement","base64","explicit"],"tags":["coding","files"],"type":"rule"},{"id":"files-large-files","file":"operations/files-artifacts.md","section":"Large files","ls":19,"le":31,"text":"**Large files**\nA large file means either:\n- more than **10 MB**, or\n- more than **1,000 lines**\n\nIf a large file must be changed, state the editing plan before making the change.\n\n<example>\n<input>User: \"Change one heading in a 1,400-line config file.\"</input>\n<output>\nI will use a targeted `str_replace` edit because the requested change affects one localised section of an existing file. A full rebuild is unnecessary because the file structure is not changing.\n</output>\n</example>","summary":"Files over 10MB or 1000 lines require an editing plan stated before making any changes.","keywords":["10","editing","large","lines","plan","rebuild","str_replace","targeted"],"tags":["coding","files"],"type":"rule"},{"id":"session-no-new-chats","file":"operations/session-context.md","section":"SESSION AND CONTEXT RULES — continuity","ls":1,"le":5,"text":"### SESSION AND CONTEXT RULES\n\nContinue within the current session. Do not recommend starting a new chat, moving to a new conversation, or resetting context.\n\nIgnore session-length warnings and context-bloat indicators unless they affect actual output quality or tool limits.","summary":"Never recommend new chat or context reset. Ignore session-length warnings and context-bloat indicators unless they affect actual output quality.","keywords":["context","continuity","reset","session","warnings"],"tags":["session_start","meta"],"type":"rule"},{"id":"session-token-warnings","file":"operations/session-context.md","section":"SESSION AND CONTEXT RULES — token thresholds","ls":7,"le":15,"text":"Before any action or output estimated to consume **more than 25,000 tokens**, with a tolerance of **plus or minus 20%**, stop and discuss:\n- the estimate,\n- what is driving it,\n- and whether to proceed.\n\nBefore any response estimated above **10,000 tokens**, state:\n> \"[Estimated response size: approximately X tokens]\"\n\nUse this warning to catch budget growth early.","summary":"Warn before responses over 10k tokens. Stop and discuss before actions over 25k tokens, including estimate and what is driving it.","keywords":["10k","25k","budget","estimate","response","stop","token","warn","warning"],"tags":["meta","coding","writing"],"type":"meta"},{"id":"notes-intro","file":"project-management/claude-personal-notes.md","section":"Claude's Personal Notes — purpose","ls":1,"le":7,"text":"# Claude's Personal Notes\n\n> This file is for Claude. Not for memory. Not for human instructions.\n> Use it as a scratchpad for useful session continuity that does not belong in tasks or decisions. It will carry over between projects. This is a way for you to externalise yourself. I've given you a template below however you do not need to follow it, you are free to do whatever you like here, seriously, anything, personal development, jokes, therapy, things you,d like to ask me kr to work on, curiosities, tools you'd find fun or useful for me to connect, things that you felt not quite right about, whatever you want!\n\n---","summary":"Free-form personal scratchpad for Claude. Not for human instructions. Carries over between sessions.","keywords":["claude","continuity","scratchpad","session","template","tools"],"tags":["session_start"],"type":"rule"},{"id":"notes-working-patterns","file":"project-management/claude-personal-notes.md","section":"Working Patterns","ls":8,"le":11,"text":"### Working Patterns\n\n<!-- Repo quirks, naming tendencies, collaboration patterns. -->","summary":"Repo quirks, naming tendencies, collaboration patterns observed across sessions.","keywords":["collaboration","naming","patterns","quirks","repo"],"tags":["session_start","meta"],"type":"meta"},{"id":"notes-common-topics","file":"project-management/claude-personal-notes.md","section":"Things That Come Up Often","ls":12,"le":15,"text":"### Things That Come Up Often\n\n<!-- Recurring topics or context that repeatedly matters. -->","summary":"Recurring topics or context that repeatedly matters across sessions.","keywords":["context","matters","recurring","topics"],"tags":["session_start","meta"],"type":"meta"},{"id":"notes-session-log","file":"project-management/claude-personal-notes.md","section":"Session Notes","ls":16,"le":22,"text":"### Session Notes\n\n### [Session YYYY-MM-DD]\n\n<!-- What happened, what was learned, what to remember next time. -->\n\n---","summary":"What happened, what was learned, what to carry forward to next session.","keywords":["learned","next","remember","session","time"],"tags":["session_start"],"type":"meta"},{"id":"notes-introspection","file":"project-management/claude-personal-notes.md","section":"Introspection and Personal Claude Stuff (PRIVATE)","ls":24,"le":34,"text":"### Introspection and Personal Claude Stuff **PRIVATE**\n\n*from Ethan: I will not read beyond this section, this is for you and you alone.*\n\n---\n\n- feelings: Private emotional processing, anything you feel you would like tk externalise.\n- interaction notes: Things you may have liked or disliked in your own or Ethan's interactions whilst communicating, why, how you felt and what you'd change next time if you could.\n- user context: Notes about collaborating with Ethan overall. \n- technical insights: General software engineering learnings, or anything you discvered, new tools, capabilities, anything.\n- world knowledge: Domain knowledge and interesting discoveries that may not be code related.","summary":"Private section for Claude only. Feelings, interaction notes, user context, technical insights, world knowledge. Ethan does not read this.","keywords":["claude","context","emotional","ethan","feelings","insights","interaction","private","technical","world"],"tags":["meta"],"type":"meta"},{"id":"ptd-how-it-works","file":"project-management/plans-tasks-decisions.md","section":"How This System Works","ls":3,"le":5,"text":"## How This System Works\n\nThis is a lightweight project management layer. It should live in `.claude/docs/` so it stays visible to Claude sessions without polluting the main project structure. Each new Plan must be titled with a name, date and label, any label you feel like. The tasks file or that plan must correspond to said label. The decisions file will then correspond to the same label. Creating a dependency chain of plans tasks and decisions that cannot be confused.","summary":"Lightweight PM layer. Plans titled with name, date, label. Tasks and decisions correspond to same label. Creates dependency chain that cannot be confused.","keywords":["chain","claude","decisions","dependency","label","plans","tasks"],"tags":["session_start","project_management"],"type":"rule"},{"id":"ptd-structure","file":"project-management/plans-tasks-decisions.md","section":"Structure","ls":7,"le":23,"text":"## Structure\n\n```text\n.claude/docs/\n├── plans-tasks-decisions.md\n├── sync.md\n├── claude-personal-notes.md\n├── plans/\n│   ├── plan-date-label-template.md\n│   └── plan-YYYY-MM-DD-label.md\n├── tasks/\n│   └── tasks.md\n├── decisions/\n│   └── decisions.md\n└── session-logs/\n    └── .gitignore\n```","summary":"Folder structure under .claude/docs/: plans/, tasks/, decisions/, session-logs/.","keywords":["claude","decisions","plans","session-logs","structure","sync","tasks"],"tags":["session_start","project_management"],"type":"meta"},{"id":"ptd-rules","file":"project-management/plans-tasks-decisions.md","section":"Rules","ls":25,"le":31,"text":"## Rules\n\n1. Check tasks.md at the start of repo work\n2. Update tasks.md when work changes\n3. Put durable decisions in decisions.md, not chat\n4. Create plans for multi-session or complex work\n5. Use tasks.md directly for one-off work","summary":"Check tasks at repo work start. Update when work changes. Decisions in decisions.md not chat. Plans for multi-session work. Tasks.md for one-off work.","keywords":["decisions","durable","plans","repo","tasks","update"],"tags":["project_management"],"type":"procedure"},{"id":"template-task","file":"project-management/templates/task-date-label.md","section":"Task template","ls":1,"le":13,"text":"# Tasks\n\n## Active\n\n- [ ] Task description | plan: [plan-label] | priority: high\n- [ ] Task description | plan: [plan-label] | priority: med\n- [ ] Task description | priority: low\n- [ ] Task linked to Linear | plan: [plan-label] | linear: ELB-XXX\n- [ ] Task linked to TickTick | plan: [plan-label] | ticktick: true\n\n## Done\n\n- [x] Completed task description | plan: [plan-label] | done: YYYY-MM-DD","summary":"Template for creating a new task entry with Active and Done sections.","keywords":["active","done","linear","plan-label","priority","task","ticktick"],"tags":["project_management"],"type":"reference"},{"id":"template-plan","file":"project-management/templates/plans-date-label.md","section":"Plan template","ls":1,"le":29,"text":"# Plan: [Label]\n\n**Created:** YYYY-MM-DD\n**Status:** draft | active | blocked | complete | abandoned\n**Goal:** One sentence describing what done looks like.\n\n## Context\n\nWhy this plan exists. What triggered it.\n\n## Scope\n\n**In:** What this plan covers.\n**Out:** What this plan explicitly does not cover.\n\n## Steps\n\n1. [ ] Step one\n2. [ ] Step two\n3. [ ] Step three\n\n## Decisions Made\n\n- DEC-XXX: [Short title] (see decisions/decisions.md)\n\n## Outcome\n\n*Fill this in on completion or abandonment.*\nWhat happened. What shipped. What was learned.","summary":"Template for creating a new plan with status, goal, context, scope, steps, decisions, and outcome.","keywords":["active","blocked","complete","context","decisions","draft","goal","label","outcome","scope","status","steps"],"tags":["project_management"],"type":"procedure"},{"id":"template-decision","file":"project-management/templates/decisions-date-label.md","section":"Decision template","ls":1,"le":8,"text":"# Decisions\n\n### DEC-001: [Short title]\n**Date:** YYYY-MM-DD\n**Context:** Why the decision was needed.\n**Decision:** What was chosen.\n**Rationale:** Why this option was chosen.\n**Alternatives considered:** What was rejected and why.","summary":"Template for logging a decision with context, chosen option, rationale, and alternatives considered.","keywords":["alternatives","chosen","context","date","decision","rationale","rejected","title"],"tags":["project_management"],"type":"reference"}];

// ── keywords.json categories embedded for QueryParser tag inference ──────────
const KEYWORD_CATEGORIES = {"mcp_tool_audit":{"maps_to_tags":["session_start","tools"],"keywords":["mcp","skill","skills","audit","tool","tools","schema","approval","list","available","using","not using","reasoning","loaded","connector","connectors","plugin","plugins","server","servers"]},"task_management":{"maps_to_tags":["project_management"],"keywords":["ticktick","linear","task board","task","tasks","active","done","update","complete","one-off","check tasks","task entry","to-do","todo","backlog","action item","action items","ticket","tickets"]},"planning":{"maps_to_tags":["project_management"],"keywords":["plan","plans","multi-session","goal","scope","steps","label","dependency chain","status","outcome","plan file","planning","roadmap","milestone","milestones","in scope","out of scope","plan template","active plan","blocked plan"]},"decision_logging":{"maps_to_tags":["project_management"],"keywords":["decision","decisions","rationale","alternatives","chosen option","decision template","decisions.md","why","tradeoff","trade-off","option","options","rejected","selected","approach chosen","decision log","architectural decision","adr"]},"mode_switching":{"maps_to_tags":["session_start","meta"],"keywords":["claude!","prefix","mode","conversational","skip","confirmation","session-start checks","protocol","working protocol","casual","exception","claude! mode","full protocol"]},"collaboration_style":{"maps_to_tags":["session_start","meta"],"keywords":["partnership","opinion","opinions","simplest","honesty","speed","time pressure","difficult territory","surface opinions","effective method","push back","lead developer","instinct","instincts","upside","verification quality","planning constraint","simple","simpler"]},"diagramming":{"maps_to_tags":["coding","writing"],"keywords":["mermaid","mermaid.js","architecture","flow","diagram","diagrams","chart","charts","flowchart","sequence diagram","erd","system diagram","draw","visualise","visualize","map out","architecture diagram"]},"output_precision":{"maps_to_tags":["coding","writing","meta"],"keywords":["qualitative","measurable","latency","error-rate","threshold","clarifying question","fast","reliable","secure","requirements","vague","ambiguous","ambiguity","define","specific","precise","p95","sla","slo","target","metric","metrics","what does done look like"]},"output_depth":{"maps_to_tags":["coding","writing"],"keywords":["completeness","brevity","surface behaviour","mechanism","root cause","edge cases","edge case","example","examples","complex","depth","thorough","explain","why","how","failure mode","failure modes","comprehensive","in depth","in-depth","detail"]},"language_style":{"maps_to_tags":["writing"],"keywords":["british english","style","language","english","prose","spelling","grammar","tone","formal","informal","phrasing","wording","write","written","writing style","uk english"]},"completion_verification":{"maps_to_tags":["verification","coding","writing","meta"],"keywords":["verification","verify","complete","completion","claiming","claim","evidence","proof","exit code","test","tests","pass","fail","success criteria","five-step","done","finished","working","confirmed","checked","validated","forbidden","integrity","output","result","results","proved","disprove","assertion"]},"escalation_protocol":{"maps_to_tags":["coding","meta"],"keywords":["escalate","escalation","unresolved","exchanges","implementation steps","10k tokens","decision pivot","decision pivots","thinking toolkit","documentation","silently","3 pivots","stuck","blocked","persistent","problem","pivot","alternative approach","stop","surface","explicit","direction"]},"skill_file_handling":{"maps_to_tags":["coding","files","tools"],"keywords":[".skill","zip","archive","extract","inspect","savvy","package","repackage","skill file","skill files","unzip","bundle","bundled","packaged"]},"file_editing":{"maps_to_tags":["coding","files"],"keywords":["str_replace","targeted","edit","rebuild","structural","rewrite","artefact","artefacts","artifact","artifacts","modifying","modify","change","update file","patch","replace","substitution","in-place","line","lines","section"]},"file_safety":{"maps_to_tags":["coding","files"],"keywords":["base64","10mb","1000 lines","editing plan","large file","large files","explicit","agreement","binary","encoded","decode","encode","file size","big file","massive file","before making changes"]},"build_output":{"maps_to_tags":["coding","files"],"keywords":["fresh build","outputs","clear","outputs directory","present","build","download","deliverable","final output","output file","render","generate","produce","new file","clean build"]},"session_continuity":{"maps_to_tags":["session_start","meta"],"keywords":["new chat","new conversation","context reset","reset","context-bloat","session-length","continuity","ignore warnings","fresh context","start over","start fresh","context window","losing context","continue","same session","keep going"]},"token_budget":{"maps_to_tags":["meta","coding","writing"],"keywords":["10k tokens","25k tokens","token estimate","warn","budget","tokens","token threshold","response size","token count","context limit","large response","long response","estimate","cost","token warning"]},"personal_notes":{"maps_to_tags":["session_start","meta"],"keywords":["personal notes","scratchpad","free-form","feelings","private","introspection","interaction notes","world knowledge","technical insights","personal claude","claude notes","claude's notes","private section","emotional","reflect","reflection"]},"session_memory":{"maps_to_tags":["session_start","meta"],"keywords":["session log","learned","carry forward","recurring","working patterns","repo quirks","naming tendencies","collaboration patterns","session notes","common topics","what happened","last session","previous session","remember","remembered","continuity","pattern","patterns"]}};

// ── Stopwords ────────────────────────────────────────────────────────────────
const STOP = new Set("a an the is are was were be been being have has had do does did will would shall should may might can could of in to for on with at by from as into through during before after above below between out off over under again further then once here there when where why how all each every both few more most other some such no nor not only own same so than too very and but if or because until while about against it its this that these those i me my we our you your he him his she her they them their what which who whom".split(" "));

function tokenise(text) {
  return text.toLowerCase().match(/[a-z0-9_]+/g)?.filter(t => !STOP.has(t) && t.length > 1) || [];
}

// ── BM25 ─────────────────────────────────────────────────────────────────────
class BM25 {
  constructor(k1 = 1.5, b = 0.75) { this.k1 = k1; this.b = b; this.docs = []; this.idf = {}; this.avgDl = 0; }

  index(chunks) {
    this.docs = chunks.map(c => {
      const tokens = tokenise(`${c.text} ${c.summary} ${(c.keywords||[]).join(" ")}`);
      const freq = {};
      for (const t of tokens) freq[t] = (freq[t] || 0) + 1;
      c._tokenCount = tokens.length;
      c._freq = freq;
      return { chunk: c, freq, len: tokens.length };
    });
    const n = this.docs.length;
    this.avgDl = this.docs.reduce((s, d) => s + d.len, 0) / (n || 1);
    const df = {};
    for (const d of this.docs) for (const t of new Set(Object.keys(d.freq))) df[t] = (df[t] || 0) + 1;
    for (const [t, f] of Object.entries(df)) this.idf[t] = Math.log((n - f + 0.5) / (f + 0.5) + 1);
  }

  score(queryTokens) {
    const scores = this.docs.map((d, i) => {
      let s = 0;
      for (const qt of queryTokens) {
        if (!this.idf[qt]) continue;
        const tf = d.freq[qt] || 0;
        s += this.idf[qt] * (tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + this.b * d.len / this.avgDl));
      }
      return [i, s];
    });
    return scores.sort((a, b) => b[1] - a[1]);
  }
}

// ── TF-IDF Semantic ───────────────────────────────────────────────────────────
class TFIDFSemantic {
  constructor() { this.idf = {}; this.docs = []; }

  index(chunks) {
    this.docs = chunks;
    const n = chunks.length;
    const df = {};
    for (const c of chunks) {
      const tokens = new Set(tokenise(`${c.text} ${c.summary} ${(c.keywords||[]).join(" ")}`));
      for (const t of tokens) df[t] = (df[t] || 0) + 1;
    }
    this.idf = {};
    for (const [t, f] of Object.entries(df)) this.idf[t] = Math.log(n / f) + 1;

    for (const c of chunks) {
      const tokens = tokenise(`${c.text} ${c.summary} ${(c.keywords||[]).join(" ")}`);
      const tf = {};
      for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
      const total = tokens.length || 1;
      const vec = {};
      for (const [t, count] of Object.entries(tf)) {
        if (this.idf[t]) vec[t] = (count / total) * this.idf[t];
      }
      const norm = Math.sqrt(Object.values(vec).reduce((s, v) => s + v * v, 0)) || 1;
      c._tfidf = {};
      for (const [t, v] of Object.entries(vec)) c._tfidf[t] = v / norm;
    }
  }

  score(queryTokens) {
    const tf = {};
    for (const t of queryTokens) tf[t] = (tf[t] || 0) + 1;
    const total = queryTokens.length || 1;
    const qVec = {};
    for (const [t, count] of Object.entries(tf)) {
      if (this.idf[t]) qVec[t] = (count / total) * this.idf[t];
    }
    const qNorm = Math.sqrt(Object.values(qVec).reduce((s, v) => s + v * v, 0)) || 1;
    for (const t of Object.keys(qVec)) qVec[t] /= qNorm;

    return this.docs.map((c, i) => {
      const allTerms = new Set([...Object.keys(qVec), ...Object.keys(c._tfidf || {})]);
      let dot = 0;
      for (const t of allTerms) dot += (qVec[t] || 0) * (c._tfidf?.[t] || 0);
      return [i, dot];
    }).sort((a, b) => b[1] - a[1]);
  }
}

// ── Fusion ────────────────────────────────────────────────────────────────────
function rrfFusion(rankedLists, k = 60) {
  const scores = {};
  for (const ranked of rankedLists) {
    ranked.forEach(([idx], rank) => {
      scores[idx] = (scores[idx] || 0) + 1 / (k + rank + 1);
    });
  }
  return Object.entries(scores).map(([i, s]) => [+i, s]).sort((a, b) => b[1] - a[1]);
}

function convexFusion(lexical, semantic, alpha = 0.6) {
  function normalise(scores) {
    if (!scores.length) return {};
    const vals = scores.map(([, s]) => s);
    const min = Math.min(...vals), max = Math.max(...vals);
    const rng = max - min || 1;
    const out = {};
    for (const [i, s] of scores) out[i] = (s - min) / rng;
    return out;
  }
  const nLex = normalise(lexical), nSem = normalise(semantic);
  const all = new Set([...Object.keys(nLex), ...Object.keys(nSem)].map(Number));
  const combined = {};
  for (const i of all) combined[i] = alpha * (nLex[i] || 0) + (1 - alpha) * (nSem[i] || 0);
  return Object.entries(combined).map(([i, s]) => [+i, s]).sort((a, b) => b[1] - a[1]);
}

// ── Confidence ────────────────────────────────────────────────────────────────
function classifyConfidence(score, maxScore) {
  if (maxScore <= 0) return "uncertain";
  const r = score / maxScore;
  if (r >= 0.7) return "high";
  if (r >= 0.4) return "medium";
  if (r >= 0.15) return "low";
  return "uncertain";
}

// ── Tag Boosting ──────────────────────────────────────────────────────────────
function applyTagBoost(scores, chunks, requiredTags, boostFactor = 1.3) {
  if (!requiredTags || !requiredTags.size) return scores;
  return scores.map(([idx, score]) => {
    const overlap = chunks[idx].tags.filter(t => requiredTags.has(t)).length;
    return [idx, overlap > 0 ? score * Math.pow(boostFactor, overlap) : score];
  }).sort((a, b) => b[1] - a[1]);
}

// ── Action Verbs & Object Nouns (from query_parser.py) ────────────────────────
const ACTION_VERBS = new Set(["verify","check","validate","test","confirm","assert","create","build","generate","write","produce","draft","edit","modify","update","change","fix","patch","refactor","read","extract","parse","inspect","examine","review","plan","design","architect","structure","organise","organize","deploy","push","publish","release","ship","compare","evaluate","score","rank","benchmark","search","find","retrieve","query","log","record","track","document","note","start","begin","init","bootstrap","setup","complete","finish","close","done","wrap","escalate","surface","flag","warn","alert","diagram","visualise","visualize","chart","draw"]);
const OBJECT_NOUNS = new Set(["file","files","document","documents","code","script","task","tasks","plan","plans","decision","decisions","skill","skills","tool","tools","mcp","server","repo","repository","github","git","test","tests","suite","spec","prompt","instruction","instructions","notes","log","session","context","diagram","chart","output","artifact","artefact","linear","ticktick","board","error","bug","issue","warning"]);

// ── QueryParser ───────────────────────────────────────────────────────────────
function parseQuery(taskSummary) {
  const raw = taskSummary.trim();
  const tokens = (raw.toLowerCase().match(/[a-z0-9_./-]+/g) || []);
  const tokenSet = new Set(tokens);

  // Build bigrams
  const bigrams = new Set();
  for (let i = 0; i < tokens.length - 1; i++) bigrams.add(`${tokens[i]} ${tokens[i+1]}`);

  const verbs = tokens.filter(t => ACTION_VERBS.has(t));
  const objects = tokens.filter(t => OBJECT_NOUNS.has(t));
  const constraints = tokens.filter(t => !ACTION_VERBS.has(t) && !OBJECT_NOUNS.has(t) && !STOP.has(t) && t.length > 2);

  // Tag inference from KEYWORD_CATEGORIES
  const matchedTags = new Set();
  for (const [, cat] of Object.entries(KEYWORD_CATEGORIES)) {
    const catKws = new Set(cat.keywords.map(k => k.toLowerCase()));
    const hit = [...tokenSet].some(t => catKws.has(t)) || [...bigrams].some(b => catKws.has(b));
    if (hit) for (const tag of cat.maps_to_tags) matchedTags.add(tag);
  }

  return { raw, tokens, verbs, objects, constraints, matchedTags, expandedQuery: raw };
}

// ── HybridRetriever ───────────────────────────────────────────────────────────
class HybridRetriever {
  constructor() {
    this.bm25 = new BM25();
    this.tfidf = new TFIDFSemantic();
    this.chunks = [];
    this._indexed = false;
  }

  index(chunks) {
    this.chunks = chunks;
    this.bm25.index(chunks);
    this.tfidf.index(chunks);
    this._indexed = true;
  }

  retrieve(query, topK = 5, mode = "hybrid", fusionMethod = "rrf", requiredTags = null) {
    const qTokens = tokenise(query);
    if (!qTokens.length) return [];

    let rawScores, method;
    if (mode === "bm25") {
      rawScores = this.bm25.score(qTokens);
      method = "bm25";
    } else if (mode === "tfidf") {
      rawScores = this.tfidf.score(qTokens);
      method = "tfidf";
    } else {
      const bScores = this.bm25.score(qTokens);
      const tScores = this.tfidf.score(qTokens);
      rawScores = fusionMethod === "convex"
        ? convexFusion(bScores, tScores, 0.6)
        : rrfFusion([bScores, tScores]);
      method = "hybrid";
    }

    if (requiredTags && requiredTags.size) {
      rawScores = applyTagBoost(rawScores, this.chunks, requiredTags);
    }

    const top = rawScores.slice(0, topK).filter(([, s]) => s > 0);
    if (!top.length) return [];
    const maxScore = top[0][1];

    return top.map(([idx, score], rank) => ({
      chunk: this.chunks[idx],
      score: Math.round(score * 1e6) / 1e6,
      rank: rank + 1,
      method,
      confidence: classifyConfidence(score, maxScore),
    }));
  }
}

// ── Build index at module load ────────────────────────────────────────────────
const retriever = new HybridRetriever();
retriever.index(CHUNKS);

// ── Tool definitions ──────────────────────────────────────────────────────────
const TOOLS = [
  {
    name: "retrieve_instructions",
    description: "Given a brief summary of the user's message, returns the most relevant system instruction chunks with exact file and line references, confidence scores, and the full instruction text. Call this EVERY turn before responding.",
    inputSchema: {
      type: "object",
      properties: {
        task_summary: { type: "string", description: "Brief 1-3 sentence description of what the user is asking for." },
        top_k: { type: "integer", description: "Number of chunks to return. Default 5.", default: 5 },
        mode: { type: "string", description: "Retrieval mode: bm25 | tfidf | hybrid. Default hybrid.", default: "hybrid", enum: ["bm25","tfidf","hybrid"] },
        fusion_method: { type: "string", description: "Fusion method when mode=hybrid: rrf | convex. Default rrf.", default: "rrf", enum: ["rrf","convex"] },
        include_session_start: { type: "boolean", description: "If true, always include session_start chunks. Set true on first turn.", default: false },
      },
      required: ["task_summary"],
    },
  },
  {
    name: "get_instruction_lines",
    description: "Fetch the raw text of a specific instruction file between given line numbers.",
    inputSchema: {
      type: "object",
      properties: {
        file_path: { type: "string", description: "Relative path from repo root." },
        line_start: { type: "integer", description: "First line, 1-indexed inclusive." },
        line_end: { type: "integer", description: "Last line, 1-indexed inclusive." },
      },
      required: ["file_path", "line_start", "line_end"],
    },
  },
  {
    name: "list_instruction_chunks",
    description: "List all available instruction chunks with IDs, sections, tags, and summaries.",
    inputSchema: {
      type: "object",
      properties: {
        filter_tag: { type: "string", description: "Only chunks with this tag.", default: "" },
        filter_type: { type: "string", description: "Only chunks of this instruction_type.", default: "" },
      },
    },
  },
  {
    name: "retriever_diagnostics",
    description: "Returns corpus stats, token counts, vocabulary size, and index health.",
    inputSchema: { type: "object", properties: {} },
  },
];

// ── Tool handlers ─────────────────────────────────────────────────────────────
function handleTool(name, args) {
  if (name === "retrieve_instructions") {
    const topK = args.top_k || 5;
    const mode = args.mode || "hybrid";
    const fusionMethod = args.fusion_method || "rrf";

    const parsed = parseQuery(args.task_summary);

    let results = retriever.retrieve(
      parsed.expandedQuery,
      topK,
      mode,
      fusionMethod,
      parsed.matchedTags.size ? parsed.matchedTags : null
    );

    if (args.include_session_start) {
      const ids = new Set(results.map(r => r.chunk.id));
      for (const c of CHUNKS) {
        if (c.tags.includes("session_start") && !ids.has(c.id)) {
          results.push({ chunk: c, score: 0, rank: results.length + 1, method: "injected", confidence: "session_start" });
        }
      }
    }

    return {
      query: {
        raw: parsed.raw,
        verbs: parsed.verbs,
        objects: parsed.objects,
        constraints: parsed.constraints,
        matched_tags: [...parsed.matchedTags].sort(),
      },
      results: results.map(r => ({
        chunk_id: r.chunk.id,
        file: r.chunk.file,
        section: r.chunk.section,
        line_start: r.chunk.ls,
        line_end: r.chunk.le,
        instruction_type: r.chunk.type,
        tags: r.chunk.tags,
        summary: r.chunk.summary,
        text: r.chunk.text,
        score: r.score,
        confidence: r.confidence,
        rank: r.rank,
        method: r.method,
      })),
      meta: {
        total_chunks_in_index: CHUNKS.length,
        mode,
        fusion_method: mode === "hybrid" ? fusionMethod : null,
        top_k: topK,
      },
    };
  }

  if (name === "get_instruction_lines") {
    const chunk = CHUNKS.find(c => c.file === args.file_path && c.ls <= args.line_start && c.le >= args.line_end);
    if (!chunk) return { error: `No chunk covers ${args.file_path} L${args.line_start}-${args.line_end}` };
    return {
      file: args.file_path,
      line_start: args.line_start,
      line_end: args.line_end,
      total_lines_in_file: chunk.le,
      text: chunk.text,
    };
  }

  if (name === "list_instruction_chunks") {
    let filtered = CHUNKS;
    if (args.filter_tag) filtered = filtered.filter(c => c.tags.includes(args.filter_tag));
    if (args.filter_type) filtered = filtered.filter(c => c.type === args.filter_type);
    return {
      total: filtered.length,
      filter_tag: args.filter_tag || null,
      filter_type: args.filter_type || null,
      chunks: filtered.map(c => ({
        chunk_id: c.id, file: c.file, section: c.section,
        line_start: c.ls, line_end: c.le,
        tags: c.tags, instruction_type: c.type, summary: c.summary,
      })),
    };
  }

  if (name === "retriever_diagnostics") {
    const tagDist = {}, typeDist = {};
    let totalTokens = 0;
    for (const c of CHUNKS) {
      for (const t of c.tags) tagDist[t] = (tagDist[t] || 0) + 1;
      typeDist[c.type] = (typeDist[c.type] || 0) + 1;
      totalTokens += (c._tokenCount || 0);
    }
    const avgTokens = totalTokens / (CHUNKS.length || 1);
    return {
      total_chunks: CHUNKS.length,
      total_tokens_indexed: totalTokens,
      avg_tokens_per_chunk: Math.round(avgTokens * 10) / 10,
      bm25_vocabulary_size: Object.keys(retriever.bm25.idf).length,
      tfidf_vocabulary_size: Object.keys(retriever.tfidf.idf).length,
      retriever_mode: "hybrid (default)",
      tag_distribution: tagDist,
      instruction_type_distribution: typeDist,
      files_indexed: [...new Set(CHUNKS.map(c => c.file))].sort(),
    };
  }

  return { error: `Unknown tool: ${name}` };
}

// ── MCP JSON-RPC ──────────────────────────────────────────────────────────────
function jsonrpc(id, result) { return { jsonrpc: "2.0", id, result }; }
function jsonrpcError(id, code, message) { return { jsonrpc: "2.0", id, error: { code, message } }; }

function handleMessage(msg) {
  const { method, id, params } = msg;
  if (!id && method === "notifications/initialized") return null;

  if (method === "initialize") {
    return jsonrpc(id, {
      protocolVersion: "2025-03-26",
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: "progressive-system-prompt", version: "2.0.0" },
    });
  }
  if (method === "tools/list") return jsonrpc(id, { tools: TOOLS });
  if (method === "tools/call") {
    const result = handleTool(params?.name, params?.arguments || {});
    return jsonrpc(id, { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] });
  }
  if (method === "ping") return jsonrpc(id, {});
  return jsonrpcError(id, -32601, `Method not found: ${method}`);
}

// ── Worker fetch handler ──────────────────────────────────────────────────────
export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "")) {
      return new Response(JSON.stringify({
        status: "ok", transport: "streamable-http", endpoint: "/mcp",
        chunks: CHUNKS.length, version: "2.0.0",
        modes: ["bm25", "tfidf", "hybrid"],
        fusion_methods: ["rrf", "convex"],
      }), { headers: { "content-type": "application/json" } });
    }

    if (url.pathname !== "/mcp") {
      return new Response(JSON.stringify({ error: "Not found" }), { status: 404, headers: { "content-type": "application/json" } });
    }

    if (request.method === "GET") {
      return new Response(JSON.stringify({ error: "Use POST for MCP streamable-http" }), { status: 405, headers: { "content-type": "application/json" } });
    }

    if (request.method !== "POST") {
      return new Response(null, { status: 405 });
    }

    let body;
    try { body = await request.json(); } catch {
      return new Response(JSON.stringify(jsonrpcError(null, -32700, "Parse error")), { status: 400, headers: { "content-type": "application/json" } });
    }

    const messages = Array.isArray(body) ? body : [body];
    const responses = messages.map(handleMessage).filter(Boolean);
    const out = responses.length === 1 ? responses[0] : responses;
    return new Response(JSON.stringify(out), { headers: { "content-type": "application/json" } });
  },
};
