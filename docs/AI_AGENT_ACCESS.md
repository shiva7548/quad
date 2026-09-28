# How to give an AI coding agent access to edit code and develop

An AI agent can only edit this repo if GitHub lets it. Below are the three
standard ways, from simplest to most controlled. Pick **one**.

> **Security rule:** agents authenticate with tokens (like passwords). Never
> paste a token into a chat, issue, or commit. Put it only in the agent's own
> settings page or GitHub Actions secrets.

---

## Option A — Invite the agent's GitHub account as a collaborator (easiest)

1. Open **https://github.com/shiva7548/quad/settings/collaborators**
2. Click **Add people** and enter the agent's GitHub username
   (for example, this session runs as `arena-ai-coding-agent[bot]`).
3. Choose role **Write** (can push code) or **Maintain** (can also manage PRs).
4. The agent accepts the invite and can immediately clone, edit, commit, push,
   and open pull requests.

Good for: dedicated bot accounts, small teams.

## Option B — Install a GitHub App (best for hosted agent products)

Products like GitHub Copilot coding agent, Claude, Codex, or Devin work as a
GitHub App:

1. In the agent product, click **Install GitHub App / Connect GitHub**.
2. Choose **Only select repositories** → pick `shiva7548/quad`.
3. Grant **Contents: Read & write**, **Pull requests: Read & write**,
   **Issues: Read & write**, **Metadata: Read**. Nothing else is needed.
4. The agent now works on branches/PRs under its own identity.

Good for: SaaS agents, least-privilege access, audit logs per app.

## Option C — Fine-grained personal access token (for self-hosted / CLI agents)

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. **Repository access:** Only select repositories → `shiva7548/quad`.
3. **Permissions:** Contents (Read and write), Pull requests (Read and write),
   Issues (Read and write), Metadata (Read). Never "All repositories".
4. Short expiry (30–90 days), then paste the token into the **agent's own
   settings / environment variable** (`GH_TOKEN`), never into chat or code.

Good for: running an agent on your own machine or server.

---

## Protect `main` so agents develop safely

Recommended once more than one writer exists:

1. **Settings → Branches → Add branch protection rule** for `main`.
2. Enable **Require a pull request before merging** and
   **Require status checks** (when CI exists).
3. Agents push feature branches and open PRs; humans (or a reviewer agent)
   approve and merge. Nothing an agent does can wipe `main` by accident.

## Workflow the agent will follow

```
edit code on branch  →  git commit  →  git push origin <branch>  →  open PR  →  review & merge
```

For this Arena session the GitHub connection is already configured
(authenticated as `arena-ai-coding-agent[bot]`), so you can simply ask the
agent to make changes — it will commit and push to its own branch
`arena/01a0e637-quad` and can open a pull request for you to review.
