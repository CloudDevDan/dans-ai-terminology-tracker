# Dan’s AI Terminology Tracker

Welcome to **Dan’s AI Terminology Tracker** — an open, visual, and structured reference map of key concepts, terms, and technologies in Artificial Intelligence and Machine Learning.

This resource is built using [**Markmap**](https://markmap.js.org/), a JavaScript-powered tool that turns Markdown lists into **interactive mind maps**. The result is a dynamic, explorable way to understand and organize terminology across a broad AI landscape - from foundational learning paradigms to Microsoft-specific AI services.

[![View the Interactive Map](https://img.shields.io/badge/View%20AI%20Mind%20Map-Live-blue?style=for-the-badge)](https://clouddevdan.github.io/dans-ai-terminology-tracker/){:target="_blank"}

---

## 🎯 About This Project

As I continue my own journey studying AI, I’ve created this tracker to:
- Serve as a **living reference point**
- Support my ongoing content series on **Azure AI Foundry**
- Help others in the **Microsoft AI community** navigate terminology more easily

🔗 **Check out the series here:**  
[daniel.mcloughlin.cloud/series/azureai](https://daniel.mcloughlin.cloud/series/azureai)

---

## 🧠 What You'll Find

This terminology tracker includes:
- Core concepts in AI, ML, and Deep Learning
- Generative AI topics like LLMs, Prompt Engineering, and RAG
- MLOps & GenAIOps terms and processes
- Data pipelines, model lifecycles, and infrastructure
- Microsoft-centric AI tools and services from Azure AI
- Related domains including NLP, CV, and Speech
- Ethics, safety, and responsible AI considerations

> **Note:** This tracker has a **Microsoft slant** — it reflects the language and concepts I encounter via **Microsoft Learn** and **Azure AI Services**.

---

## 📈 Visual Format

All terminology is structured in a Markdown file, which is rendered visually using **Markmap**.  
You can:
- View it interactively in the browser
- Navigate through the hierarchy
- Expand/collapse terms as needed

Stay tuned for a live link to the hosted Markmap page, or run it locally using the CLI.

---

## 🤝 Community Contributions

If you’re part of the Microsoft AI community, or just exploring AI terminology like I am, feel free to:
- Suggest new terms or definitions
- Submit pull requests for corrections or additions
- Help grow the visual map

---

## 🪪 License

This project is licensed under the **MIT License** — you're free to use, share, remix, and build upon it, as long as attribution is given.

---

## 🛠 How It's Built

This project uses [Markmap](https://markmap.js.org) to convert a Markdown list into an interactive mind map.  
After exporting the map, a custom **post-processing script** applies visual enhancements including:

- Dark mode theme
- Custom font and header
- White text and dark background for readability
- A favicon
- Branding and blog link

---

## ⚙️ Setup Instructions

Install the Markmap CLI (if not already installed):

```bash
npm install -g markmap-cli
```

Then run this command from the root of the repo:

```bash
npx markmap-cli markmap.md -o ./docs/index.html && node postprocess-map.js
```

This generates the visual map and applies styling in one go.

---

## 🌐 Why the `/docs` Folder?

The `docs` folder is used because GitHub Pages can be configured to serve static content from it.  
This means your `index.html` is published at:

📍 `https://clouddevdan.github.io/dans-ai-terminology-tracker/index.html`