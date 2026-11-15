MemEye canonical image payloads live in task-specific subdirectories under `data/image/`.

Use:

- `data/image/Brand_Memory_Test/...`
- `data/image/Chat_UI_Memory_Test/...`
- `data/image/ComicScene_Alley_Oop_Draft/...`
- `data/image/Home_Renovation_Interior_Design/...`
- `data/image/Visual_Case_Archive_Assistant/...`

Do not place benchmark images at the repository root.

Naming rules:

- keep the image directory name identical to the matching JSON filename stem
- prefer underscores over spaces in canonical task names
- reference images from dialog JSON as `../image/<Task_Name>/<file>`
