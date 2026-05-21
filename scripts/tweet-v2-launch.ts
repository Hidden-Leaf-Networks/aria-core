/**
 * Aria Core v2.0.0 Launch Tweet
 * Uses media-kit-skill for image gen + text copy + x-skill for posting
 *
 * Run: npx tsx scripts/tweet-v2-launch.ts
 * Requires: OPENAI_API_KEY, X_* env vars
 */

import 'dotenv/config';
import { resolve } from 'path';

const MEDIA_KIT_PATH = resolve(__dirname, '../../media-kit-skill');
const X_SKILL_PATH = resolve(__dirname, '../../x-skill');

async function main() {
  // Dynamic imports
  const mediaKit = await import(`${MEDIA_KIT_PATH}/dist/index.js`);
  const xSkill = await import(`${X_SKILL_PATH}/dist/index.js`);

  const imageGen = mediaKit.createImageGeneratorFromEnv();
  const textGen = mediaKit.createTextGeneratorFromEnv();
  const xClient = xSkill.createXClientFromEnv();

  // --- Generate Image ---
  console.log('🎨 Generating launch image...');
  const imageResult = await imageGen.generate({
    template: 'product-launch',
    params: {
      productName: 'Aria Core v2.0.0',
      tagline: 'Open-source AI agent framework with multi-tenant white-label, MCP + A2A, and deterministic execution.',
      features: [
        'MCP Server + Client',
        'A2A Protocol',
        'OpenTelemetry Tracing',
        'Visual Workflow Editor',
        'RAG Knowledge System',
        'Agent Marketplace',
        'Voice Pipeline',
        'Deep Bridge Consensus',
      ],
      callToAction: 'github.com/Hidden-Leaf-Networks/aria-core',
      version: 'v2.0.0',
    },
  });
  console.log(`  Image: ${imageResult.path}`);

  // --- Generate Tweet Copy ---
  console.log('✍️  Generating tweet copy...');
  const input = {
    template: 'product-launch' as const,
    params: {
      productName: 'Aria Core v2.0.0',
      tagline: 'The only open-source AI agent framework with multi-tenant white-label SaaS',
      features: [
        'Deterministic 8-state FSM — no uncontrolled loops',
        'Deep Bridge multi-model consensus — unique in market',
        'MCP Server + Client — 97M+ SDK downloads standard',
        'A2A Protocol — Google-backed, 50+ enterprise partners',
        'OpenTelemetry tracing — no LangSmith lock-in',
        'Visual Workflow Editor with React Flow',
        'RAG / Knowledge with tenant-scoped retrieval',
        'Agent Marketplace for community sharing',
        'Voice Pipeline (Whisper + TTS)',
        'Agentic Data Cloud query federation',
        'Per-tenant Stripe billing built in',
        '371 tests, Apache 2.0 licensed',
      ],
      competitors: 'LangGraph Cloud has no multi-tenant or white-label. CrewAI Enterprise charges $60-120k/yr. Neither has Deep Bridge consensus or deterministic risk scoring.',
      callToAction: 'github.com/Hidden-Leaf-Networks/aria-core',
      version: 'v2.0.0',
    },
  };

  // Generate 3 tweets for a thread
  const tweet1Copy = await textGen.generate(input, 'x-thread-hook', 'ai-engineering');
  const tweet2Copy = await textGen.generate(
    { ...input, params: { ...input.params, tagline: 'How Aria Core stacks up vs LangGraph and CrewAI' } },
    'x-thread-body',
    'ai-engineering',
  );
  const tweet3Copy = await textGen.generate(
    { ...input, params: { ...input.params, tagline: 'Full feature list for v2.0.0' } },
    'x-thread-close',
    'ai-engineering',
  );

  console.log('\n📝 Generated Thread:');
  console.log('--- Tweet 1 ---');
  console.log(tweet1Copy.text);
  console.log('--- Tweet 2 ---');
  console.log(tweet2Copy.text);
  console.log('--- Tweet 3 ---');
  console.log(tweet3Copy.text);

  // --- Confirm before posting ---
  const readline = await import('readline');
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const answer = await new Promise<string>((resolve) => {
    rl.question('\n🚀 Post this thread? (y/n): ', resolve);
  });
  rl.close();

  if (answer.toLowerCase() !== 'y') {
    console.log('Cancelled.');
    return;
  }

  // --- Upload & Post ---
  console.log('📤 Uploading image...');
  const mediaId = await xClient.uploadMedia(imageResult.path);

  console.log('🐦 Posting thread...');
  const result1 = await xClient.postTweet(tweet1Copy.text, { mediaIds: [mediaId] });
  console.log(`  Tweet 1: ${result1.id}`);

  const result2 = await xClient.postTweet(tweet2Copy.text, { replyTo: result1.id });
  console.log(`  Tweet 2: ${result2.id}`);

  const result3 = await xClient.postTweet(tweet3Copy.text, { replyTo: result2.id });
  console.log(`  Tweet 3: ${result3.id}`);

  console.log(`\n✅ Thread live: https://x.com/HiddenLeafHQ/status/${result1.id}`);
}

main().catch(console.error);
