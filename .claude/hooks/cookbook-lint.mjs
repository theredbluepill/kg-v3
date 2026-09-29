#!/usr/bin/env node
/**
 * Cross-harness PostToolUse lint for cookbook notes — the universal subset.
 *
 * Validates frontmatter, the six required fields, known types, and provenance for a
 * claimed Decision adoption. It handles single-file writes and every cookbook target in
 * an apply_patch payload. Bundle-specific register checks stay in the bundle's own copy.
 * Shape is not truth: a well-typed fabricated number still passes.
 */
import { readFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { isAbsolute, relative, resolve, sep } from 'node:path'

// Git supplies staged bytes; an unstaged repair cannot conceal a bad commit.
if (process.argv.includes('--staged-sources')) {
  try {
    const git = (...args) => execFileSync('git', args, { encoding: 'utf8' })
    const paths = git('diff', '--cached', '--name-only', '-z').split('\0')
    const notes = git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').split('\0').filter(isConcept)
    const problems = notes.flatMap((path) =>
      sourceProblems(git('show', `:${path}`)).map((problem) => `${path}: ${problem}`))
    if (process.argv.includes('--require-log') && paths.some(isConcept) && !paths.includes('cookbook/log.md')) {
      problems.unshift('cookbook notes staged without cookbook/log.md; record the change in the same commit')
    }
    if (problems.length) {
      process.stderr.write(`cookbook source references:\n${problems.join('\n')}\n`)
      process.exit(1)
    }
    process.exit(0)
  } catch (error) {
    process.stderr.write(`cookbook source references: ${error.message}\n`)
    process.exit(1)
  }
}

let payload
try {
  payload = JSON.parse(readFileSync(0, 'utf8'))
} catch {
  allow()
}

const root = resolve(
  process.env.COOKBOOK_ROOT
    ?? process.env.CLAUDE_PROJECT_DIR
    ?? payload?.cwd
    ?? process.cwd(),
)
const reports = editTargets(payload?.tool_name, payload?.tool_input)
  .map(lintTarget)
  .filter(Boolean)
if (reports.length === 0) allow()

const message = [
  'cookbook-lint:',
  ...reports.flatMap(({ target, problems }) => [
    `- ${target}`,
    ...problems.map((problem) => `  - ${problem}`),
  ]),
  'Fix the note now, in this turn — shape drift compounds silently.',
].join('\n')
output({
  hookSpecificOutput: {
    hookEventName: 'PostToolUse',
    additionalContext: message,
  },
})

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
  if (target === '' || target.startsWith('../')) return ''
  return target
}

function isConcept(target) {
  if (!target.startsWith('cookbook/') || !target.endsWith('.md')) return false
  const basename = target.split('/').pop().replace(/\.md$/u, '')
  return basename !== 'index' && basename !== 'log'
}

// Inspect declared sources only, not prose, examples or unrelated metadata.
// Block and flow lists use single-line resource scalars; aliases/multiline
// resources receive a diagnostic rather than an inferred value.
function sourceProblems(text) {
  const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/u.exec(text)?.[1]
  if (!frontmatter) return []
  const startsOfSources = [...frontmatter.matchAll(/^(?:sources|"sources"|'sources'):[ \t]*/gmu)]
  if (startsOfSources.length > 1) return ['sources: duplicate declarations; keep one explicit source list']
  const start = startsOfSources[0]
  if (!start) return []
  let block = frontmatter.slice(start.index + start[0].length)
  const end = block.search(/\n(?:[A-Za-z_][\w-]*|"[^"\n]+"|'[^'\n]+'):[ \t]*/u)
  if (end >= 0) block = block.slice(0, end)
  block = block.replace(/^(?:\s*#[^\n]*(?:\n|$))+/u, '')
  if (/^\s*[&*!|>]/u.test(block)) return ['sources: use an explicit block or flow list; source-list aliases/anchors are unsupported']
  const resources = []
  const problems = []
  if (/^\s*[\[{]/u.test(block)) {
    resources.push(...flowResources(block))
  } else {
    const lines = block.split(/\r?\n/u)
    const entries = lines.map((line, index) => ({ index, match: /^( *)-(?:([ \t]+)(.*))?$/u.exec(line) }))
      .filter(({ match }) => match)
    const indent = Math.min(...entries.map(({ match }) => match[1].length))
    const starts = entries.filter(({ match }) => match[1].length === indent)
    starts.forEach(({ index, match }, entry) => {
      const rest = lines.slice(index + 1, starts[entry + 1]?.index ?? lines.length)
      const first = match[3] ?? ''
      if (/^[&*!|>]/u.test(first)) {
        problems.push('sources: use explicit source entries; source-entry aliases/anchors are unsupported')
        return
      }
      if (first.startsWith('{')) {
        resources.push(...flowResources([first, ...rest].join('\n')))
        return
      }
      const content = rest.filter((line) => line.trim() && !line.trimStart().startsWith('#'))
      const width = !first.trim() || first.trimStart().startsWith('#')
        ? Math.min(...content.map((line) => /^ */u.exec(line)[0].length))
        : indent + 1 + (match[2]?.length ?? 1)
      const direct = [first, ...rest.filter((line) =>
        Number.isFinite(width) && line.startsWith(' '.repeat(width)) && !/^\s/u.test(line.slice(width)))
        .map((line) => line.slice(width))]
      for (const line of direct) {
        if (/^(?:<<|"<<"|'<<'):[ \t]*/u.test(line)) problems.push('sources: merge keys are unsupported; use explicit source entries')
        const field = /^(?:resource|"resource"|'resource'):[ \t]*(.*)$/u.exec(line)
        if (field) resources.push(field[1])
      }
    })
  }
  return [...problems, ...resources.flatMap((raw) => {
    const { value, error } = sourceScalar(raw)
    const problem = error ?? resourceProblem(value, raw)
    return problem ? [`sources[].resource ${JSON.stringify(value ?? raw)}: ${problem}`] : []
  })]
}

function quotedEnd(text, start) {
  const quote = text[start]
  for (let index = start + 1; index < text.length; index++) {
    if (quote === '"' && text[index] === '\\') { index++; continue }
    if (text[index] !== quote) continue
    if (quote === "'" && text[index + 1] === "'") { index++; continue }
    return index + 1
  }
  return text.length
}

function flowResources(text) {
  const resources = []
  const keys = []
  const objectArrays = []
  const directArray = text.trimStart().startsWith('[') ? 1 : 0
  let arrays = 0
  for (let index = 0; index < text.length;) {
    const char = text[index]
    if (/\s/u.test(char)) { index++; continue }
    if (char === '#') {
      const newline = text.indexOf('\n', index)
      index = newline < 0 ? text.length : newline + 1
      continue
    }
    if (char === '{') { keys.push(true); objectArrays.push(arrays); index++; continue }
    if (char === '}') { keys.pop(); objectArrays.pop(); index++; continue }
    if (char === ',') {
      if (keys.length && arrays === objectArrays.at(-1)) keys[keys.length - 1] = true
      index++; continue
    }
    if (char === ':') { if (keys.length) keys[keys.length - 1] = false; index++; continue }
    if (char === '[') { arrays++; index++; continue }
    if (char === ']') { arrays--; index++; continue }
    const end = char === '"' || char === "'" ? quotedEnd(text, index)
      : index + (/^[^\s{}\[\],:]+/u.exec(text.slice(index))?.[0].length ?? 1)
    const key = text.slice(index, end).replace(/^["']|["']$/gu, '')
    const after = /^\s*:\s*/u.exec(text.slice(end))
    if (arrays === directArray && ((keys.length === 0 && after)
        || (keys.length <= 1 && /^[&*!]/u.test(char) && (keys.length === 0 || keys[0]))
        || (keys.length === 1 && keys[0] && key === '<<' && after))) {
      resources.push('*unsupported-source-entry')
    }
    if (keys.length === 1 && arrays === directArray && keys[0] && key === 'resource' && after) {
      const begin = end + after[0].length
      const finish = text[begin] === '"' || text[begin] === "'" ? quotedEnd(text, begin)
        : begin + (/^[^,}\r\n]*/u.exec(text.slice(begin))?.[0].length ?? 0)
      resources.push(text.slice(begin, finish))
      keys[0] = false
      index = finish
    } else index = end
  }
  return resources
}

function sourceScalar(raw) {
  const text = raw.trim()
  if (!text || /[\r\n]/u.test(text) || /^[|>*&!{\[]/u.test(text)) return { error: 'use a single-line source string (repository:PATH for a repository source)' }
  try {
    if (text.startsWith('"')) {
      const end = quotedEnd(text, 0)
      if (!/^\s*(?:#.*)?$/u.test(text.slice(end))) throw new Error('trailing scalar text')
      return { value: JSON.parse(text.slice(0, end)) }
    }
    if (text.startsWith("'")) {
      const end = quotedEnd(text, 0)
      if (text[end - 1] !== "'" || end < 2 || !/^\s*(?:#.*)?$/u.test(text.slice(end))) throw new Error('invalid scalar')
      return { value: text.slice(1, end - 1).replace(/''/gu, "'") }
    }
    return { value: text.replace(/\s+#.*$/u, '').trimEnd() }
  } catch {
    return { error: 'use a well-formed single-line source string (repository:PATH for a repository source)' }
  }
}

function repositoryTypo(prefix) {
  const expected = 'repository'
  if (['repo', 'repos'].includes(prefix)) return true
  if (prefix.length !== expected.length) {
    const [longer, shorter] = prefix.length > expected.length ? [prefix, expected] : [expected, prefix]
    return longer.length === shorter.length + 1
      && [...longer].some((_, i) => longer.slice(0, i) + longer.slice(i + 1) === shorter)
  }
  const changed = [...prefix].flatMap((c, i) => c === expected[i] ? [] : [i])
  return changed.length <= 1 || (changed.length === 2 && changed[1] === changed[0] + 1
    && prefix[changed[0]] === expected[changed[1]] && prefix[changed[1]] === expected[changed[0]])
}

function resourceProblem(value, raw) {
  const scheme = /^([A-Za-z][A-Za-z0-9+.-]*):/u.exec(value)?.[1]
  if (!scheme || /^[A-Za-z]:[\\/]/u.test(value)) return 'missing source namespace; use repository:PATH for a repository source'
  if (scheme !== 'repository' && repositoryTypo(scheme.toLowerCase())) return 'repository source prefix must be exactly repository:'
  if (scheme !== 'repository') return null
  const path = value.slice('repository:'.length).split('#', 1)[0]
  const parts = path.replace(/\/$/u, '').split('/')
  if (!path || path !== path.trim() || /^[~/]/u.test(path) || /[\\:\u0000-\u001f\u007f]/u.test(path)
      || parts.some((part) => !part || part === '.' || part === '..')) {
    return 'repository:PATH must use a canonical repo-relative path with forward slashes'
  }
  if (!raw.includes(value)) return 'write the repository:PATH literally; encoded escapes do not match the read-first gate'
  return null
}

function lintTarget(target) {
  if (!isConcept(target)) return null

  let text
  try {
    text = readFileSync(resolve(root, target), 'utf8')
  } catch {
    return null
  }

  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---/u)
  const problems = sourceProblems(text)
  const standingBoard = /^cookbook\/decisions\/the-.+-board\.md$/u.test(target)
  if (standingBoard && text.length > 12_000) {
    problems.push('standing-board maintenance: exceeds the 12,000-character budget. '
      + 'Keep current position, ranked options and tombstones; preserve history in linked '
      + 'notes/log before shortening. Do not auto-truncate or change decisions to fit.')
  }
  if (standingBoard && hasLeadingProseTab(text)) {
    problems.push('standing-board maintenance: leading tab indentation outside fenced code '
      + 'can turn prose into a code block. Use consistent spaces for prose/list continuation.')
  }
  if (!match) {
    problems.push('note has no frontmatter block')
  } else {
    const frontmatter = match[1]
    const has = (field) => new RegExp(`^${field}:`, 'mu').test(frontmatter)
    for (const field of ['type', 'title', 'description', 'tags', 'status', 'generated']) {
      if (!has(field)) problems.push(`missing required field \`${field}\``)
    }
    const type = (frontmatter.match(/^type:\s*"?([A-Za-z-]+)"?/mu) ?? [])[1]
    const known = [
      'Experiment',
      'Episode',
      'Decision',
      'Reference',
      'Workflow',
      'Lesson',
      'Observation',
      'Register',
    ]
    if (type && !known.includes(type)) {
      problems.push(`type "${type}" is not one of ${known.join('/')}`)
    }
    if (type === 'Decision') {
      const quoted = /^(decider|ratified|adopted_by):/mu.test(frontmatter)
        || /^>/mu.test(text.slice(match[0].length))
      if (!quoted && /adopted by|approved by|owner('s)? (?:call|directive)/iu.test(text)) {
        problems.push('Decision note claims an adoption/directive but has no `decider:`/'
          + '`ratified:` field and no blockquote of the decider\'s words — a decision quotes '
          + 'its decider')
      }
    }
  }
  return problems.length > 0 ? { target, problems } : null
}

function hasLeadingProseTab(text) {
  let fence = ''
  for (const line of text.split('\n')) {
    const marker = /^ {0,3}(`{3,}|~{3,})(.*)$/u.exec(line)
    if (fence) {
      if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length
        && marker[2].trim() === '') fence = ''
      continue
    }
    if (marker && (marker[1][0] === '~' || !marker[2].includes('`'))) {
      fence = marker[1]
      continue
    }
    if (/^[ \t]*\t/u.test(line)) return true
  }
  return false
}

function allow() {
  output({})
}

function output(value) {
  process.stdout.write(JSON.stringify(value))
  process.exit(0)
}
