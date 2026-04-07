# Deploy to HuggingFace Spaces — Exact Commands

## One-time setup (5 minutes)

### 1. Get your HF token
Go to: https://huggingface.co/settings/tokens
Click "New token" → name it → select **Write** scope → Create → Copy it

### 2. Run these commands

```bash
cd er_triage_env

# Set your token
export HF_TOKEN=hf_YOUR_TOKEN_HERE

# Deploy (replace with your HF username)
openenv push --repo-id YOUR_USERNAME/er-triage-env
```

That's it. openenv push will:
1. Authenticate with HF using your token
2. Create the Space automatically (Docker SDK)
3. Upload all files
4. Trigger a build

## After deploy (~2-3 min for build)

Your Space URL: `https://YOUR_USERNAME-er-triage-env.hf.space`

Validate it:
```bash
openenv validate https://YOUR_USERNAME-er-triage-env.hf.space
```

Should show: 6/6 criteria passed

## Step 6 — Submit

Paste this URL into the hackathon form:
```
https://YOUR_USERNAME-er-triage-env.hf.space
```

## Optional: set HF Space secrets (for LLM inference)
In your Space → Settings → Variables and secrets:
- `OPENAI_API_KEY` = your OpenAI key
- `MODEL_NAME` = gpt-4o-mini
- `ER_TASK` = easy
