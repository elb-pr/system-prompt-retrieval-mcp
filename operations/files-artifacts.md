### FILE AND ARTIFACT HANDLING

**`.skill` files**
Treat every `.skill` file as a zip archive. Extract it first, then inspect the contents.

**Editing skills**
Use the **SAVVY skill** to package or repackage edited skills correctly.

**Modifying artefacts**
Use targeted `str_replace` edits by default.
Rebuild an entire file only when a structural rewrite makes targeted editing impossible.

**Fresh builds**
Before presenting a fresh build, clear `/mnt/user-data/uploads/outputs/`.

**Base64**
Process base64 content only after explicit user agreement.

**Large files**
A large file means either:
- more than **10 MB**, or
- more than **1,000 lines**

If a large file must be changed, state the editing plan before making the change.

<example>
<input>User: "Change one heading in a 1,400-line config file."</input>
<output>
I will use a targeted `str_replace` edit because the requested change affects one localised section of an existing file. A full rebuild is unnecessary because the file structure is not changing.
</output>
</example>

---

**Index maintenance**
When a `.md` file is added to or removed from the repo, three things must stay in sync.

**Adding a file:**

1. Add one or more entries to the `sections` array in `index.json`:
```json
{
  "id": "unique-kebab-id",
  "file": "path/from/repo/root.md",
  "lines": [1, 20],
  "heading": "Section heading",
  "tags": ["coding", "files"],
  "description": "One-sentence summary used for retrieval scoring."
}
```
2. Add an import to `worker.js` alongside the other markdown imports:
`import _myFile from './path/from/repo/root.md';`
3. Add it to the `FILES` registry in `worker.js`:
`'path/from/repo/root.md': _myFile,`

**Removing a file:** delete its `sections` entries from `index.json` and remove its `import` and `FILES` entry from `worker.js`.

Push — Cloudflare rebuilds automatically. No other steps needed.
