# Secrets and Environment Variables Configuration

This document details all required secrets and environment variables for the Enterprise Policy RAG system.

## GitHub Secrets

Configure these secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

### Required for CI/CD

| Secret Name | Description | Example |
|------------|-------------|---------|
| `GITHUB_TOKEN` | Automatically provided by GitHub Actions | Auto-generated |
| `RENDER_API_KEY` | Render API key for deployment | `rnd_xxx...` |
| `RENDER_SERVICE_ID` | Render service ID | `srv-xxx...` |

### How to Get Render Credentials

1. **RENDER_API_KEY**:
   - Go to [Render Dashboard](https://dashboard.render.com/account/api-keys)
   - Click "Create API Key"
   - Copy the generated key

2. **RENDER_SERVICE_ID**:
   - Go to your service dashboard on Render
   - The Service ID is in the URL: `https://dashboard.render.com/web/srv-XXXXXX`
   - Or run: `curl -H "Authorization: Bearer YOUR_API_KEY" https://api.render.com/v1/services`

## Application Environment Variables

Configure these in your deployment platform (Render, Docker, etc.):

### Core Application Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `APP_NAME` | No | "Enterprise Policy RAG" | Application name |
| `ENVIRONMENT` | No | "Dev" | Environment (Dev/Staging/Prod) |
| `LOG_LEVEL` | No | "INFO" | Logging level (DEBUG/INFO/WARNING/ERROR) |

### API Keys

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_API_KEY` | Yes | - | HuggingFace API key for embeddings |
| `LLM_API_KEY` | Yes | - | API key for LLM (Groq/OpenAI/etc.) |
| `LLM_API_URL` | No | - | Custom LLM API endpoint URL |

#### How to Get API Keys

**HuggingFace API Key:**
1. Sign up at [HuggingFace](https://huggingface.co/)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with read access

**Groq API Key:**
1. Sign up at [Groq](https://console.groq.com/)
2. Go to [API Keys](https://console.groq.com/keys)
3. Create a new API key

### Storage Configuration (Optional)

For persistent FAISS index storage using S3-compatible object storage:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_BUCKET` | No | - | S3 bucket name for FAISS index |
| `S3_ENDPOINT` | No | - | S3 endpoint URL (for non-AWS S3) |
| `S3_INDEX_KEY` | No | "faiss.index" | S3 object key for index file |
| `S3_META_KEY` | No | "metadata.json" | S3 object key for metadata file |
| `AWS_ACCESS_KEY_ID` | No | - | AWS/S3 access key ID |
| `AWS_SECRET_ACCESS_KEY` | No | - | AWS/S3 secret access key |
| `AWS_REGION` | No | "us-east-1" | AWS region |

#### Storage Providers

**AWS S3:**
- Use standard AWS credentials
- Set `AWS_REGION` appropriately
- No need to set `S3_ENDPOINT`

**DigitalOcean Spaces:**
```env
S3_ENDPOINT=https://nyc3.digitaloceanspaces.com
AWS_ACCESS_KEY_ID=your_spaces_key
AWS_SECRET_ACCESS_KEY=your_spaces_secret
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
```

**Cloudflare R2:**
```env
S3_ENDPOINT=https://your-account.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=your_r2_key
AWS_SECRET_ACCESS_KEY=your_r2_secret
S3_BUCKET=your-bucket-name
```

### FAISS Index Paths

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FAISS_INDEX_PATH` | No | "/data/processed/faiss.index" | Local path to FAISS index |
| `FAISS_META_PATH` | No | "/data/processed/metadata.json" | Local path to metadata |

## Render Configuration

### Setting Environment Variables in Render

1. Go to your service in [Render Dashboard](https://dashboard.render.com/)
2. Navigate to "Environment" tab
3. Add each environment variable with its value
4. Click "Save Changes"

### Sample Render Environment Configuration

```yaml
# Minimal configuration
HF_API_KEY: hf_xxxxxxxxxxxxx
LLM_API_KEY: gsk_xxxxxxxxxxxxx
LLM_API_URL: https://api.groq.com/openai/v1

# With S3 storage (recommended for production)
HF_API_KEY: hf_xxxxxxxxxxxxx
LLM_API_KEY: gsk_xxxxxxxxxxxxx
S3_BUCKET: my-rag-index-bucket
S3_ENDPOINT: https://nyc3.digitaloceanspaces.com
AWS_ACCESS_KEY_ID: DO00xxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY: xxxxxxxxxxxxxxxxx
```

## Docker Deployment

### Using Environment Variables with Docker

```bash
docker run -p 8000:8000 \
  -e HF_API_KEY=your_hf_key \
  -e LLM_API_KEY=your_llm_key \
  -e LLM_API_URL=https://api.groq.com/openai/v1 \
  -e S3_BUCKET=your-bucket \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  ghcr.io/your-username/enterprise-policy-rag:latest
```

### Using .env File with Docker

Create a `.env` file:
```env
HF_API_KEY=hf_xxxxxxxxxxxxx
LLM_API_KEY=gsk_xxxxxxxxxxxxx
LLM_API_URL=https://api.groq.com/openai/v1
S3_BUCKET=my-bucket
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

Run with:
```bash
docker run -p 8000:8000 --env-file .env ghcr.io/your-username/enterprise-policy-rag:latest
```

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use different API keys** for dev/staging/production
3. **Rotate secrets regularly** (every 90 days recommended)
4. **Use least privilege** - only grant necessary permissions
5. **Enable MFA** on cloud provider accounts
6. **Monitor API usage** for unusual activity
7. **Use secret scanning** tools in CI/CD
8. **Encrypt secrets at rest** when storing backups

## Troubleshooting

### Common Issues

**"FAISS index missing" error:**
- Upload a PDF through `/api/upload` endpoint
- Or configure S3 storage with pre-existing index
- Check `FAISS_INDEX_PATH` and `FAISS_META_PATH` settings

**"S3 download failed" warning:**
- Verify S3 credentials are correct
- Check bucket name and endpoint URL
- Ensure bucket permissions allow read/write
- Confirm index files exist in S3

**"API key invalid" errors:**
- Verify API keys are active and valid
- Check for extra spaces or newlines in key values
- Ensure keys have necessary permissions
- Test keys with provider's API directly

**GitHub Actions deployment fails:**
- Verify `RENDER_API_KEY` secret is set correctly
- Check `RENDER_SERVICE_ID` matches your service
- Review Render service logs for errors
- Ensure Render service is not suspended

## Contact & Support

For issues or questions:
- Open an issue in the GitHub repository
- Check Render status page for outages
- Review application logs in Render dashboard
