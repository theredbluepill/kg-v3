#!/usr/bin/env node
/**
 * Cross-harness Stop hook: correction-shaped numeric claims must survive a record check.
 *
 * Codex supplies `last_assistant_message`; Claude Code supplies a transcript path. The hook
 * blocks once and asks the agent to cite or retract the correction after checking the bundle.
 * It recognizes the shape of a correction, not whether the claim is true.
 */
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

let payload
try {
  payload = JSON.parse(readFileSync(0, 'utf8'))
} catch {
  allow()
}
if (payload?.stop_hook_active) allow()

const lastText = typeof payload?.last_assistant_message === 'string'
  ? payload.last_assistant_message
  : lastClaudeAssistantText(payload?.agent_transcript_path ?? payload?.transcript_path)
if (!lastText) allow()

if (!hasNumericCorrection(lastText)) allow()

const markers = join(tmpdir(), 'cookbook-correction-check')
const marker = join(
  markers,
  createHash('sha256')
    .update(`${payload?.session_id ?? ''}\u0000${lastText}`)
    .digest('hex')
    .slice(0, 24),
)
if (existsSync(marker)) allow()
try {
  mkdirSync(markers, { recursive: true })
  writeFileSync(marker, '1')
} catch {
  allow()
}

output({
  decision: 'block',
  reason: 'cookbook-correction-check: this reply corrects a factual claim involving numbers. '
    + 'Before finishing, grep the cookbook (log.md and notes) for recorded observations that '
    + 'bear on the corrected fact, then cite the supporting record or retract/relabel the claim '
    + 'as inference. If already verified this turn, restate the citation in one line and finish.',
})

function hasNumericCorrection(text) {
  // Match within a sentence/paragraph, not against unrelated measurements elsewhere.
  // Keep single wrapped newlines and decimal points inside the same claim.
  const claims = text.split(/[。！？;；!?]|\.(?!\d)|\r?\n[ \t]*\r?\n/u)
  const correction = /(其實是|其實不是|我錯了|糾正|勘誤|\bcorrected\b|actually (?:is|starts|was)|intel (?:was|is) wrong)/iu
  // A bare “不是…而是…” is often rhetorical, not a numeric correction. Require
  // numbers in the contrasted values themselves rather than elsewhere in the claim.
  const contrast = /不是\s*(\S{1,12}?)\s*[,，]?\s*(?:而)?是\s*([^\n，,]{1,24})/giu
  return claims.some((claim) => (
    (correction.test(claim) && /\d/u.test(claim))
      || [...claim.matchAll(contrast)].some((match) => /\d/u.test(match[1]) && /\d/u.test(match[2]))
  ))
}

function lastClaudeAssistantText(transcriptPath) {
  if (typeof transcriptPath !== 'string' || !existsSync(transcriptPath)) return ''
  try {
    const lines = readFileSync(transcriptPath, 'utf8').trim().split('\n')
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      let entry
      try {
        entry = JSON.parse(lines[index])
      } catch {
        continue
      }
      if (entry?.type !== 'assistant') continue
      const content = entry?.message?.content
      if (!Array.isArray(content)) continue
      const text = content
        .filter((item) => item?.type === 'text')
        .map((item) => item.text)
        .join('\n')
      if (text) return text
    }
  } catch {
    return ''
  }
  return ''
}

function allow() {
  output({})
}

function output(value) {
  writeFileSync(1, JSON.stringify(value))
  process.exit(0)
}
