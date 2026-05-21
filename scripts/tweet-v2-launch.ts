/**
 * Aria Core v2.0.0 Launch Tweet Thread
 *
 * Uses media-kit-skill for image gen + OAuth 1.0a for posting (OAuth2 token expired).
 * Posts a 3-tweet thread to @HiddenLeafHQ.
 *
 * Run from aria-core root:
 *   NODE_PATH="/home/tre/Projects/hidden-leaf/x-skill/node_modules:/home/tre/Projects/hidden-leaf/media-kit-skill/node_modules" npx tsx scripts/tweet-v2-launch.ts
 */

import { resolve, dirname } from 'path';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { config } from 'dotenv';
import * as crypto from 'crypto';
import axios from 'axios';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load env from sibling repos
const X_SKILL_PATH = resolve(__dirname, '../../x-skill');
const MEDIA_KIT_PATH = resolve(__dirname, '../../media-kit-skill');

config({ path: resolve(MEDIA_KIT_PATH, '.env') });
config({ path: resolve(X_SKILL_PATH, '.env') });

// ── OAuth 1.0a Signing ─────────────────────────────────────────────────────

const pEnc = (s: string) =>
  encodeURIComponent(s).replace(/[!'()*]/g, (c) => '%' + c.charCodeAt(0).toString(16).toUpperCase());

function oauth1Sign(method: string, url: string, bodyParams?: Record<string, string>): string {
  const consumerKey = process.env.X_ORG_CONSUMER_KEY!;
  const consumerSecret = process.env.X_ORG_CONSUMER_SECRET!;
  const accessToken = process.env.X_ORG_OAUTH1_ACCESS_TOKEN!;
  const accessTokenSecret = process.env.X_ORG_OAUTH1_ACCESS_TOKEN_SECRET!;

  const oauthParams: Record<string, string> = {
    oauth_consumer_key: consumerKey,
    oauth_nonce: crypto.randomBytes(16).toString('hex'),
    oauth_signature_method: 'HMAC-SHA1',
    oauth_timestamp: Math.floor(Date.now() / 1000).toString(),
    oauth_token: accessToken,
    oauth_version: '1.0',
  };

  // For JSON body requests, only oauth params go in the signature base
  const allParams = { ...oauthParams, ...(bodyParams || {}) };
  const sorted = Object.keys(allParams)
    .sort()
    .map((k) => pEnc(k) + '=' + pEnc(allParams[k]))
    .join('&');
  const baseString = method + '&' + pEnc(url) + '&' + pEnc(sorted);
  const signingKey = pEnc(consumerSecret) + '&' + pEnc(accessTokenSecret);
  oauthParams.oauth_signature = crypto.createHmac('sha1', signingKey).update(baseString).digest('base64');

  return (
    'OAuth ' +
    Object.keys(oauthParams)
      .sort()
      .map((k) => pEnc(k) + '="' + pEnc(oauthParams[k]) + '"')
      .join(', ')
  );
}

async function postTweetOAuth1(text: string, options?: { mediaIds?: string[]; replyToId?: string }): Promise<{ id: string; url: string }> {
  const url = 'https://api.twitter.com/2/tweets';
  const body: any = { text };
  if (options?.mediaIds?.length) {
    body.media = { media_ids: options.mediaIds };
  }
  if (options?.replyToId) {
    body.reply = { in_reply_to_tweet_id: options.replyToId };
  }

  // For JSON body POST, body params are NOT included in OAuth signature
  const authHeader = oauth1Sign('POST', url);

  const resp = await axios.post(url, body, {
    headers: {
      Authorization: authHeader,
      'Content-Type': 'application/json',
    },
  });

  const id = resp.data?.data?.id;
  if (!id) throw new Error('Tweet post failed: ' + JSON.stringify(resp.data));
  return { id, url: `https://x.com/HiddenLeafHQ/status/${id}` };
}

// ── Tweet Copy ──────────────────────────────────────────────────────────────

const TWEET_1 = `Aria Core v2.0.0 is live.

The only open-source AI agent framework with multi-tenant white-label SaaS.

Deterministic 8-state FSM
Deep Bridge multi-model consensus
MCP Server + Client | A2A Protocol
371 tests | Apache 2.0

github.com/Hidden-Leaf-Networks/aria-core`;

const TWEET_2 = `How Aria Core stacks up:

vs LangGraph Cloud:
→ We have multi-tenant, white-label, risk scoring, A2A
→ They don't

vs CrewAI Enterprise:
→ Deterministic FSM, event sourcing, $0-499/mo
→ They charge $60-120k/yr

Open source wins.`;

const TWEET_3 = `Full v2.0.0 feature list:

▸ MCP Server + Client
▸ A2A Protocol
▸ OpenTelemetry Tracing
▸ Visual Workflow Editor
▸ RAG Knowledge System
▸ Agent Marketplace
▸ Voice Pipeline
▸ Agentic Data Cloud

Built by @HiddenLeafHQ`;

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  // Dynamic import for media-kit-skill
  const mediaKit = await import(`${MEDIA_KIT_PATH}/dist/index.js`);
  // x-skill for media upload only (uses OAuth 1.0a internally)
  const xSkill = await import(`${X_SKILL_PATH}/dist/index.js`);

  // ── Generate Image ──────────────────────────────────────────────────────
  let imageBuffer: Buffer | null = null;
  try {
    console.log('Generating launch image...');
    const imageGen = mediaKit.createImageGeneratorFromEnv();
    const imageResult = await imageGen.generate({
      template: 'product-launch',
      format: 'og',
      productName: 'Aria Core v2.0.0',
      tagline: 'Open-source AI agent framework with multi-tenant white-label, MCP + A2A, and deterministic execution.',
      features: [
        'Multi-tenant White-label SaaS',
        'MCP + A2A + Deep Bridge Consensus',
        '371 Tests | Apache 2.0 Licensed',
      ] as [string, string, string],
      version: 'v2.0.0',
    });
    console.log(`  Image saved: ${imageResult.path}`);
    imageBuffer = readFileSync(imageResult.path);
  } catch (err: any) {
    console.warn(`Image generation failed: ${err.message}`);
    console.warn('Proceeding with text-only thread.');
  }

  // ── Upload Media (uses x-skill's OAuth 1.0a upload) ─────────────────────
  let mediaId: string | undefined;
  if (imageBuffer) {
    try {
      console.log('Uploading image to X...');
      // Use x-skill's XClient just for media upload (OAuth 1.0a, doesn't need OAuth2)
      const XClient = xSkill.XClient;
      // Create a minimal client — we only need uploadMedia which uses OAuth 1.0a directly
      const oauth1Config = {
        consumerKey: process.env.X_ORG_CONSUMER_KEY!,
        consumerSecret: process.env.X_ORG_CONSUMER_SECRET!,
        accessToken: process.env.X_ORG_OAUTH1_ACCESS_TOKEN!,
        accessTokenSecret: process.env.X_ORG_OAUTH1_ACCESS_TOKEN_SECRET!,
      };
      // uploadMedia is a static-like method that only uses OAuth 1.0a
      // But it's an instance method, so we need a client instance
      // Since OAuth2 is broken, create with a dummy token and only use uploadMedia
      const dummyClient = new XClient({
        accessToken: 'dummy',
        userId: process.env.X_ORG_USER_ID!,
      });
      const uploadResult = await dummyClient.uploadMedia(imageBuffer, oauth1Config);
      mediaId = uploadResult.mediaId;
      console.log(`  Media ID: ${mediaId}`);
    } catch (err: any) {
      console.warn(`Media upload failed: ${err.message}`);
      console.warn('Proceeding without image.');
    }
  }

  // ── Print Thread ────────────────────────────────────────────────────────
  console.log('\n--- Tweet 1 (hook + image) ---');
  console.log(TWEET_1);
  console.log(`  [${TWEET_1.length} chars]`);
  console.log(`\n--- Tweet 2 (competitive) ---`);
  console.log(TWEET_2);
  console.log(`  [${TWEET_2.length} chars]`);
  console.log(`\n--- Tweet 3 (features) ---`);
  console.log(TWEET_3);
  console.log(`  [${TWEET_3.length} chars]`);
  console.log(`\nImage: ${mediaId ? 'attached (' + mediaId + ')' : 'none'}`);

  // ── Post Thread via OAuth 1.0a ──────────────────────────────────────────
  console.log('\nPosting thread via OAuth 1.0a...');

  try {
    // Tweet 1
    const opts1: any = {};
    if (mediaId) opts1.mediaIds = [mediaId];
    const result1 = await postTweetOAuth1(TWEET_1, opts1);
    console.log(`  Tweet 1 posted: ${result1.url}`);

    // Tweet 2 — reply to tweet 1
    const result2 = await postTweetOAuth1(TWEET_2, { replyToId: result1.id });
    console.log(`  Tweet 2 posted: ${result2.url}`);

    // Tweet 3 — reply to tweet 2
    const result3 = await postTweetOAuth1(TWEET_3, { replyToId: result2.id });
    console.log(`  Tweet 3 posted: ${result3.url}`);

    console.log(`\nThread live: ${result1.url}`);
  } catch (err: any) {
    console.error(`\nPosting failed: ${err.message}`);
    if (err.response?.status === 403) {
      console.error('\n403 Forbidden — likely causes:');
      console.error('  1. X API spend cap reached (check developer.x.com billing)');
      console.error('  2. Free tier 280 char limit (check tweet lengths above)');
      console.error('  3. App permissions — need Read+Write');
    }
    if (err.response?.data) {
      console.error('Response data:', JSON.stringify(err.response.data, null, 2));
    }
    throw err;
  }
}

main().catch((err) => {
  console.error('Fatal:', err.message);
  process.exit(1);
});
