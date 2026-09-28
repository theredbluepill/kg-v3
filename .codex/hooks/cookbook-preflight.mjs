#!/usr/bin/env node
/**
 * Cross-harness cookbook preflight for Claude Code and Codex.
 *
 * SessionStart/UserPromptSubmit/SubagentStart restore bounded bundle navigation and
 * decision context. Small standing boards enter in full; large boards become a compact
 * projection and remain available for an explicit read. PreToolUse refuses the first edit,
 * per session and target, when a cookbook note cites that target as a `repository:` source.
 * Editing the cookbook itself is the act of recording the reading.
 */
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from 'node:fs'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'

let payload
try {
  payload = JSON.parse(readFileSync(0, 'utf8'))
} catch {
  allow()
}

const event = payload?.hook_event_name
const root = repositoryRoot(
  process.env.COOKBOOK_ROOT
    ?? process.env.CLAUDE_PROJECT_DIR
    ?? payload?.cwd
    ?? process.cwd(),
)
const cookbook = join(root, 'cookbook')
const markerRoot = join(
  tmpdir(),
  'cookbook-read-first',
  createHash('sha256').update(root).digest('hex').slice(0, 16),
)
const unguarded = new Set(['cookbook', '.git', 'node_modules'])
const contextEvents = new Set([
  'SessionStart',
  'UserPromptSubmit',
  'SubagentStart',
])

if (contextEvents.has(event)) injectContext()
else if (event === 'PreToolUse') gateEdit()
else allow()

function repositoryRoot(start) {
  const cwd = resolve(start)
  try {
    return resolve(execFileSync('git', ['rev-parse', '--show-toplevel'], {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim())
  } catch {
    return cwd
  }
}

function injectContext() {
  const index = join(cookbook, 'index.md')
  const log = join(cookbook, 'log.md')
  if (!existsSync(index) || !existsSync(log)) allow()

  const paths = [index, log, ...standingBoards()]
  const sections = []
  for (const path of paths) {
    let raw
    try {
      raw = readFileSync(path, 'utf8')
    } catch {
      continue
    }
    const name = relative(root, path).split(sep).join('/')
    const text = path === log
      ? raw.split('\n').slice(0, 140).join('\n')
      : path === index
        ? indexNavigation(raw)
        : standingBoardContext(raw, name)
    sections.push(`--- BEGIN ${name} ---\n${text}\n--- END ${name} ---`)
  }
  if (sections.length === 0) allow()

  output({
    hookSpecificOutput: {
      hookEventName: event,
      additionalContext: sections.join('\n\n'),
    },
  })
}

function standingBoardContext(raw, name) {
  const fullBoardMaximum = 12_000
  if (raw.length <= fullBoardMaximum) return raw

  const { body, identity } = boardParts(raw)
  const { preamble, sections } = boardSections(body)
  const position = sections.find(({ heading }) => /\bposition\b/iu.test(heading))
  const ranked = sections.find(({ heading }) => (
    /\b(?:open (?:options|levers|moves)|ranked|follow[- ]?ups?|next moves?)\b/iu.test(heading)
  ))
  const endgame = sections.find(({ heading }) => /\bendgame\b/iu.test(heading))
  const dead = sections.find(({ heading }) => (
    /\b(?:dead lanes?|tombstones?|known limits?)\b/iu.test(heading)
  ))
  const selected = new Set([position, ranked, endgame, dead].filter(Boolean))
  const fallback = sections.filter((section) => !selected.has(section))
  const parts = [
    '[compact standing-board projection]',
    `Full board: ${name}. Read it before opening an episode or relying on omitted history.`,
    '[standing-board maintenance: exceeds the 12,000-character budget. '
      + 'Projection is a context fallback, not a healthy board. Reconcile current position, '
      + 'rankings and tombstones; preserve history in linked notes/log before shortening. '
      + 'Do not auto-truncate or treat omitted evidence as superseded.]',
    identity,
    clipHead(preamble, 3_200, 'board preamble'),
    position
      ? clipHeadTail(position.text, 4_000, 'position')
      : clipHeadTail(fallback.shift()?.text ?? '', 4_000, 'current section'),
    ranked
      ? clipHead(ranked.text, 3_500, 'ranked options')
      : clipHead(fallback.shift()?.text ?? '', 3_500, 'open options'),
    endgame ? clipHead(endgame.text, 1_500, 'endgame') : '',
    dead ? tombstoneLeads(dead.text, 1_700) : '',
  ].filter((part) => part.trim() !== '')
  const projection = parts.join('\n\n').trimEnd()
  const projectionMaximum = 15_800
  if (projection.length <= projectionMaximum) return projection
  return [
    projection.slice(0, projectionMaximum - 90).trimEnd(),
    '',
    '[board projection hard-capped; read the full board at the path above]',
  ].join('\n')
}

function boardParts(raw) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/u.exec(raw)
  if (!match) return { body: raw, identity: '' }
  const scalarKeys = new Set(['type', 'title', 'description', 'status'])
  const identityLines = match[1].split('\n').filter((line) => {
    const key = /^([a-z_]+):/u.exec(line)?.[1]
    return key && scalarKeys.has(key)
  })
  return {
    body: raw.slice(match[0].length),
    identity: identityLines.length > 0
      ? ['---', ...identityLines, '---'].join('\n')
      : '',
  }
}

function boardSections(body) {
  const lines = body.split('\n')
  const starts = []
  lines.forEach((line, index) => {
    if (/^##\s+\S/u.test(line)) starts.push(index)
  })
  const first = starts[0] ?? lines.length
  const sections = starts.map((start, index) => {
    const end = starts[index + 1] ?? lines.length
    const selected = lines.slice(start, end).join('\n').trimEnd()
    return {
      heading: lines[start].replace(/^##\s+/u, '').trim(),
      text: selected,
    }
  })
  return {
    preamble: lines.slice(0, first).join('\n').trimEnd(),
    sections,
  }
}

function clipHead(text, maximum, label) {
  if (!text || text.length <= maximum) return text
  const marker = `\n\n[${label} excerpt truncated; read the full board]`
  const limit = maximum - marker.length
  const boundary = text.lastIndexOf('\n', limit)
  return text.slice(0, boundary > 0 ? boundary : limit).trimEnd() + marker
}

function clipHeadTail(text, maximum, label) {
  if (!text || text.length <= maximum) return text
  const marker = `\n\n[${label} middle omitted; read the full board]\n\n`
  const available = maximum - marker.length
  const headLimit = Math.floor(available * 0.35)
  const tailLimit = available - headLimit
  const headBoundary = text.lastIndexOf('\n', headLimit)
  const tailStart = Math.max(0, text.length - tailLimit)
  const tailBoundary = markdownBlockStart(text, tailStart)
  const head = text.slice(0, headBoundary > 0 ? headBoundary : headLimit).trimEnd()
  const tail = text.slice(tailBoundary).trimStart()
  return head + marker + tail
}

function markdownBlockStart(text, minimum) {
  const lines = text.split('\n')
  let offset = 0
  let previousBlank = true
  for (const line of lines) {
    const nonblank = line.trim() !== ''
    const topLevelBlock = /^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)/u.test(line)
    if (offset >= minimum && nonblank && (previousBlank || topLevelBlock)) return offset
    previousBlank = !nonblank
    offset += line.length + 1
  }
  return text.length
}

function tombstoneLeads(text, maximum) {
  const lines = text.split('\n')
  const heading = lines[0] ?? '## Dead lanes'
  const items = []
  let item = []
  for (const line of lines.slice(1)) {
    if (/^[-*+]\s+\S/u.test(line)) {
      if (item.length > 0) items.push(item)
      item = [line]
    } else if (item.length > 0) {
      item.push(line)
    }
  }
  if (item.length > 0) items.push(item)

  const titles = items.flatMap((itemLines) => {
    const compact = itemLines.join(' ').replace(/\s+/gu, ' ').trim()
    const boldTitle = /^[-*+]\s+(\*\*[\s\S]*?\*\*)/u.exec(compact)?.[1]
    return boldTitle ? [`* ${boldTitle}`] : []
  })
  const full = [heading, '', ...titles].join('\n')
  if (full.length <= maximum) return full

  const marker = '\n\n[tombstone titles truncated; read the full board]'
  let bounded = heading
  for (const title of titles) {
    const separator = bounded === heading ? '\n\n' : '\n'
    if (bounded.length + separator.length + title.length + marker.length > maximum) break
    bounded += separator + title
  }
  return bounded.trimEnd() + marker
}

function indexNavigation(raw) {
  const lines = raw.split('\n')
  const browse = lines.findIndex((line) => line.trim() === '# Browse')
  if (browse >= 0) {
    const nextSection = lines.findIndex(
      (line, index) => index > browse && /^#\s+\S/u.test(line),
    )
    const end = nextSection >= 0 ? nextSection : lines.length
    return lines.slice(0, end).join('\n').trimEnd()
  }

  const maximumLines = 80
  if (lines.length <= maximumLines) return raw
  return [
    ...lines.slice(0, maximumLines),
    '',
    '[index navigation truncated after 80 lines]',
  ].join('\n')
}

function standingBoards() {
  const decisions = join(cookbook, 'decisions')
  if (!existsSync(decisions)) return []
  try {
    return readdirSync(decisions, { withFileTypes: true })
      .filter((entry) => entry.isFile() && /^the-.+-board\.md$/u.test(entry.name))
      .map((entry) => join(decisions, entry.name))
      .sort()
  } catch {
    return []
  }
}

function gateEdit() {
  if (!existsSync(cookbook)) allow()
  const governed = editTargets(payload?.tool_name, payload?.tool_input)
    .filter((target) => target && !unguarded.has(target.split('/')[0]))
    .map((target) => ({ target, notes: governingNotes(target) }))
    .filter(({ notes }) => notes.length > 0)
  if (governed.length === 0) allow()

  const session = String(payload?.session_id ?? 'no-session')
  const unread = governed.filter(({ target }) => !existsSync(markerFor(session, target)))
  if (unread.length === 0) allow()

  for (const { target } of unread) {
    try {
      mkdirSync(markerRoot, { recursive: true })
      writeFileSync(markerFor(session, target), `${target}\n`)
    } catch {
      // Without a marker the next attempt blocks again, which is safer than silent bypass.
    }
  }

  deny([
    'Read the governing cookbook note(s), then submit the same change again.',
    '',
    ...unread.flatMap(({ target, notes }) => [
      `  target: ${target}`,
      ...notes.flatMap((note) => [
        `    ${note.path}`,
        `      ${note.heading}`,
        ...note.directives.map((item) => `      standing directive: ${item}`),
      ]),
    ]),
    '',
    'Recorded project memory is part of the change contract, not optional chat context.',
  ].join('\n'))
}

function editTargets(toolName, toolInput) {
  const name = String(toolName ?? '')
  const serialized = JSON.stringify(toolInput ?? {})
  const mutating = /(?:apply[_-]?patch|edit|write|multiedit)/iu.test(name)
    || serialized.includes('*** Begin Patch')
  if (!mutating) return []

  const found = []
  collectFilePaths(toolInput, found)
  for (const match of serialized.matchAll(
    /\*\*\* (?:Add|Update|Delete) File: ([^\\\n"]+)/gu,
  )) {
    found.push(match[1])
  }
  return [...new Set(found.map(normalizeTarget).filter(Boolean))]
}

function collectFilePaths(value, found) {
  if (!value || typeof value !== 'object') return
  if (typeof value.file_path === 'string') found.push(value.file_path)
  for (const child of Object.values(value)) {
    if (child && typeof child === 'object') collectFilePaths(child, found)
  }
}

function normalizeTarget(path) {
  const absolute = isAbsolute(path) ? resolve(path) : resolve(root, path)
  const target = relative(root, absolute).split(sep).join('/')
  return target === '' || target.startsWith('../') ? '' : target
}

function governingNotes(target) {
  const notes = []
  for (const path of markdownFiles(cookbook)) {
    let text
    try {
      text = readFileSync(path, 'utf8')
    } catch {
      continue
    }
    if (!text.includes(`repository:${target}`)) continue
    const lines = text.split('\n')
    const directives = []
    lines.forEach((line, index) => {
      if (!line.includes('user-directive:')) return
      for (const candidate of lines.slice(index, index + 3)) {
        const title = /title:\s*"([^"]+)"/u.exec(candidate)
        if (title) {
          directives.push(title[1])
          break
        }
      }
    })
    notes.push({
      path: relative(root, path).split(sep).join('/'),
      heading: lines.find((line) => line.startsWith('# '))?.slice(2).trim()
        ?? relative(root, path),
      directives: [...new Set(directives)],
    })
  }
  return notes
}

function markdownFiles(directory) {
  const found = []
  let entries
  try {
    entries = readdirSync(directory, { withFileTypes: true })
  } catch {
    return found
  }
  for (const entry of entries) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) found.push(...markdownFiles(path))
    else if (entry.name.endsWith('.md')) found.push(path)
  }
  return found
}

function markerFor(session, target) {
  return join(
    markerRoot,
    createHash('sha256').update(`${session}\u0000${target}`).digest('hex'),
  )
}

function deny(reason) {
  output({
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: 'deny',
      permissionDecisionReason: reason,
    },
  })
}

function allow() {
  output({})
}

function output(value) {
  writeFileSync(1, JSON.stringify(value))
  process.exit(0)
}
