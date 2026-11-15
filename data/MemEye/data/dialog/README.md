MemEye canonical dialog payloads live in this directory.

Each dataset should use:

- `data/dialog/<Task_Name>.json`
- `data/image/<Task_Name>/...`

Current benchmark-facing JSON schema:

```json
{
  "character_profile": { "...": "..." },
  "multi_session_dialogues": [
    {
      "session_id": "D1",
      "date": "2026-03-01",
      "dialogues": [
        {
          "round": "D1:1",
          "user": "...",
          "assistant": "...",
          "input_image": ["../image/<Task_Name>/<file>.png"]
        }
      ]
    }
  ],
  "human-annotated QAs": [
    {
      "question": "...",
      "answer": "...",
      "point": [["X2"], ["Y1"]],
      "session_id": ["D1"],
      "clue": ["D1:1"]
    }
  ]
}
```

Canonical rules:

- keep task names in filenames and image directories aligned
- use underscores, not spaces, in canonical task directory names
- store benchmark answers directly in the QA object, not only in hidden metadata
- use MemEye `point=[[X...],[Y...]]` coordinates rather than ad hoc skill tags
- keep `clue` values aligned with actual round ids in `multi_session_dialogues`
