import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { Check, CircleHelp, Command, Copy, Crown, FileText, Link2, Menu, Moon, Plus, Reply, Search, SendHorizontal, Share2, Sparkles, Sun, ThumbsDown, ThumbsUp, Trash2, UserMinus, Users, X } from 'lucide-react'
import Particles, { ParticlesProvider } from '@tsparticles/react'
import { loadSlim } from '@tsparticles/slim'
import MarkdownResponse from './components/MarkdownResponse.jsx'
import './App.css'

const starterMessages = [
  {
    id: 1,
    role: 'assistant',
    text: "I'm online and ready. What can I help you with?",
    time: 'Now',
    createdAt: new Date().toISOString(),
  },
]

const suggestions = [
  { label: 'Compare ChatGPT, Gemini, and Claude.', icon: Search },
  { label: 'What do you know about me?', icon: Sparkles },
  { label: 'What are the good restaurants near me?', icon: FileText },
  { label: 'How is the weather at my place?', icon: Command },
]

const shareHistoryOptions = [
  { value: 'all', label: 'Share all previous conversation' },
  { value: 'past_3_days', label: 'Share past 3 days conversation' },
  { value: 'none', label: 'Do not share conversation' },
]

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')
const AUTH_TOKEN_KEY = 'nexa.auth.token'
const PENDING_INVITE_KEY = 'nexa.pendingInvite.token'
const PENDING_GOOGLE_SERVICE_KEY = 'nexa.pendingGoogle.service'
const THEME_KEY = 'nexa.theme'
const LOCATION_CACHE_PREFIX = 'nexa.location.'
const LOCATION_CACHE_TTL = 1000 * 60 * 30
const MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
const AGENT_PREFIX_PATTERN = /^@nexa(?:\b|$)[\s,:-]*/i
const AGENT_DRAFT_PATTERN = /^@(?:n(?:e(?:x(?:a)?)?)?)(?:\b|$)[\s,:-]*/i
const AGENT_PARTIAL_PATTERN = /^@(?:n(?:e(?:x(?:a)?)?)?)?$/i
const RESEARCH_COMMAND_PATTERN = /^\/research(?:\s|$)/i
const GOOGLE_SERVICE_DEFAULTS = [
  { service: 'gmail', label: 'Gmail', configured: true, connected: false, email: '' },
  { service: 'google_calendar', label: 'Google Calendar', configured: true, connected: false, email: '' },
  { service: 'google_drive', label: 'Google Drive', configured: true, connected: false, email: '' },
]

const locationRequestPattern = /\b(?:near me|around me|nearby|closest|nearest|from me|where am i|my location|directions?|route|distance|how far|how long|travel time|away|local|near my|restaurants?|cafes?|coffee shops?|hotels?|attractions?|pharmacies|hospitals?|gas stations?|petrol pumps?|atms?|parking|weather|forecast|traffic|places?)\b/i
const LOCATION_PERMISSION_REQUIRED_MESSAGE = `Location permission is required to process this query. Kindly provide access and try again.

Browser steps:
1. Click the location or lock icon beside the address bar.
2. Set Location to Allow for Nexa.
3. Refresh the page, then ask the location query again.

If it still does not work, open browser Site settings, allow Location for this site, and make sure device location services are turned on.`

function readAuthToken() {
  try {
    return window.localStorage.getItem(AUTH_TOKEN_KEY) || ''
  } catch {
    return ''
  }
}

function writeAuthToken(token = '') {
  try {
    if (token) window.localStorage.setItem(AUTH_TOKEN_KEY, token)
    else window.localStorage.removeItem(AUTH_TOKEN_KEY)
  } catch {
    // Auth still falls back to cookies when local storage is unavailable.
  }
}

function readThemePreference() {
  try {
    const stored = window.localStorage.getItem(THEME_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // Theme still works for this session.
  }
  return 'light'
}

function authHeaders(headers = {}) {
  const token = readAuthToken()
  return token ? { ...headers, Authorization: `Bearer ${token}` } : headers
}

function apiFetch(resource, options = {}) {
  return fetch(resource, {
    ...options,
    headers: authHeaders(options.headers || {}),
  })
}

function inviteTokenFromPath(pathname = window.location.pathname) {
  return /^\/join\/([^/]+)$/.exec(pathname)?.[1] || ''
}

function readPendingInviteToken() {
  try {
    return window.sessionStorage.getItem(PENDING_INVITE_KEY) || ''
  } catch {
    return ''
  }
}

function writePendingInviteToken(token = '') {
  try {
    if (token) window.sessionStorage.setItem(PENDING_INVITE_KEY, token)
    else window.sessionStorage.removeItem(PENDING_INVITE_KEY)
  } catch {
    // Invite links still work when the token remains in the URL.
  }
}

function readPendingGoogleService() {
  try {
    return window.sessionStorage.getItem(PENDING_GOOGLE_SERVICE_KEY) || ''
  } catch {
    return ''
  }
}

function writePendingGoogleService(service = '') {
  try {
    if (service) window.sessionStorage.setItem(PENDING_GOOGLE_SERVICE_KEY, service)
    else window.sessionStorage.removeItem(PENDING_GOOGLE_SERVICE_KEY)
  } catch {
    // OAuth still works; the UI just cannot preserve the interim status.
  }
}

function mergeGoogleServices(services = []) {
  const incoming = new Map((services || []).map((service) => [service.service, service]))
  const merged = GOOGLE_SERVICE_DEFAULTS.map((fallback) => ({
    ...fallback,
    ...(incoming.get(fallback.service) || {}),
  }))
  for (const service of services || []) {
    if (!GOOGLE_SERVICE_DEFAULTS.some((fallback) => fallback.service === service.service)) merged.push(service)
  }
  return merged
}

function googleIconForService(service = '') {
  if (service === 'google_calendar') return 'calendar.png'
  if (service === 'google_drive') return 'drive.png'
  return 'gmail.png'
}

function compactGoogleLabel(label = '') {
  return String(label || '').replace(/^Google\s+/i, '')
}

function formatMessageTime(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(date.getTime())) return 'Just now'
  return new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  }).format(date)
}

function hasAgentMention(value = '') {
  return AGENT_PREFIX_PATTERN.test(String(value).trimStart())
}

function stripAgentMention(value = '') {
  return String(value).trimStart().replace(AGENT_PREFIX_PATTERN, '')
}

function stripAgentDraftMention(value = '') {
  return String(value).trimStart().replace(AGENT_DRAFT_PATTERN, '')
}

function startOfLocalDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

function formatMessageDay(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const today = startOfLocalDay(new Date())
  const target = startOfLocalDay(date)
  const dayOffset = Math.round((today - target) / 86400000)
  if (dayOffset === 0) return 'Today'
  if (dayOffset === 1) return 'Yesterday'
  return new Intl.DateTimeFormat(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  }).format(date)
}

function messageWithDocumentReference(message, id) {
  const content = String(message?.content || '')
  const createdAt = message.created_at || new Date().toISOString()
  const replyTo = message?.reply_to ? {
    id: String(message.reply_to.id || ''),
    role: String(message.reply_to.role || ''),
    text: String(message.reply_to.content || ''),
    senderName: String(message.reply_to.sender_name || ''),
    sender_user_id: String(message.reply_to.sender_user_id || ''),
  } : null
  const marker = /\n\s*\nDocument:\s*/i
  const match = marker.exec(content)
  if (!match) {
    return {
      id,
      role: message.role,
      text: content,
      time: formatMessageTime(createdAt),
      createdAt,
      senderName: message.sender_name,
      sender_user_id: message.sender_user_id,
      target_user_id: message.target_user_id,
      targetEmail: message.target_email,
      systemAction: message.system_action,
      researchRunId: message.research_run_id || '',
      feedback: message.feedback || '',
      ...(replyTo ? { replyTo } : {}),
    }
  }
  const documentName = content
    .slice(match.index + match[0].length)
    .split(/\s+Document:\s*/i)[0]
    .trim()
  return {
    id,
    role: message.role,
    text: content.slice(0, match.index).trim(),
    documentName,
    time: formatMessageTime(createdAt),
    createdAt,
    senderName: message.sender_name,
    sender_user_id: message.sender_user_id,
    target_user_id: message.target_user_id,
    targetEmail: message.target_email,
    systemAction: message.system_action,
    researchRunId: message.research_run_id || '',
    feedback: message.feedback || '',
    ...(replyTo ? { replyTo } : {}),
  }
}

function areMessagesEquivalent(current, next) {
  if (current.length !== next.length) return false
  return current.every((message, index) => {
    const candidate = next[index]
    return String(message.id) === String(candidate.id)
      && message.role === candidate.role
      && message.text === candidate.text
      && message.time === candidate.time
      && message.createdAt === candidate.createdAt
      && message.senderName === candidate.senderName
      && message.sender_user_id === candidate.sender_user_id
      && message.target_user_id === candidate.target_user_id
      && message.targetEmail === candidate.targetEmail
      && message.systemAction === candidate.systemAction
      && message.researchRunId === candidate.researchRunId
      && message.feedback === candidate.feedback
      && message.documentName === candidate.documentName
      && JSON.stringify(message.cards || null) === JSON.stringify(candidate.cards || null)
      && JSON.stringify(message.replyTo || null) === JSON.stringify(candidate.replyTo || null)
  })
}

function areSessionsEquivalent(current, next) {
  if (current.length !== next.length) return false
  return current.every((session, index) => {
    const candidate = next[index]
    return session.id === candidate.id
      && session.title === candidate.title
      && session.updated_at === candidate.updated_at
      && session.shared === candidate.shared
      && session.member_count === candidate.member_count
      && Number(session.unread_count || 0) === Number(candidate.unread_count || 0)
  })
}

function replyPreviewFromMessage(message) {
  if (!message || message.role === 'system') return null
  const text = String(message.text || '').replace(/\s+/g, ' ').trim()
  return {
    id: String(message.id),
    role: message.role,
    text: text.slice(0, 180),
    senderName: message.senderName || '',
    sender_user_id: message.sender_user_id || '',
  }
}

function userLocationCacheKey(user) {
  const identity = String(user?.id || user?.email || '').trim()
  return identity ? `${LOCATION_CACHE_PREFIX}${identity}` : ''
}

function readCachedLocation(user) {
  const key = userLocationCacheKey(user)
  if (!key) return null
  try {
    const cached = JSON.parse(window.localStorage.getItem(key) || 'null')
    if (!cached || cached.status !== 'granted' || !cached.location) return cached
    const freshEnough = Date.now() - Number(cached.updatedAt || 0) < LOCATION_CACHE_TTL
    return freshEnough ? cached : { ...cached, stale: true }
  } catch {
    return null
  }
}

function writeCachedLocation(user, value) {
  const key = userLocationCacheKey(user)
  if (!key) return
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Location still works for this session even when local storage is disabled.
  }
}

function currentBrowserLocation() {
  if (!window.isSecureContext) {
    return Promise.resolve({
      location: null,
      error: 'Browser location requires http://127.0.0.1:8000, http://localhost:8000, or HTTPS. Open Nexa locally and try again.',
    })
  }
  if (!navigator.geolocation) {
    return Promise.resolve({
      location: null,
      error: 'This browser does not provide location services. Use Chrome or Edge and allow location access.',
    })
  }
  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => resolve({
      location: null,
      error: 'Location request timed out. Check that Windows location services are enabled and try again.',
    }), 7000)
    navigator.geolocation.getCurrentPosition(
      (position) => {
        window.clearTimeout(timeout)
        resolve({
          location: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
        })
      },
      (error) => {
        window.clearTimeout(timeout)
        const message = error?.code === error?.PERMISSION_DENIED
          ? 'Location permission is blocked for Nexa. Click the location icon beside the browser address bar, allow location, then retry.'
          : error?.code === error?.POSITION_UNAVAILABLE
            ? 'Your browser could not determine a location. Enable Windows location services and try again.'
            : 'Location request timed out. Check that Windows location services are enabled and try again.'
        resolve({ location: null, error: message })
      },
      { enableHighAccuracy: false, maximumAge: 300000, timeout: 6500 },
    )
  })
}

function parseMarkdownTables(text = '') {
  const lines = text.split('\n')
  const tables = []
  for (let index = 0; index < lines.length - 1; index += 1) {
    const headerLine = lines[index].trim()
    const dividerLine = lines[index + 1].trim()
    if (!headerLine.includes('|') || !/^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(dividerLine)) {
      continue
    }

    const headers = splitMarkdownRow(headerLine)
    const rows = []
    let cursor = index + 2
    while (cursor < lines.length && lines[cursor].includes('|') && lines[cursor].trim()) {
      const cells = splitMarkdownRow(lines[cursor])
      if (cells.length === headers.length) rows.push(cells)
      cursor += 1
    }
    if (headers.length > 1 && rows.length) {
      tables.push({ headers, rows })
      index = cursor - 1
    }
  }
  return tables.slice(0, 3)
}

function splitMarkdownRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.replace(/\*\*/g, '').replace(/`/g, '').trim())
}

function numericValue(value) {
  const cleaned = String(value || '').replace(/[$,%]/g, '').replace(/,/g, '').trim()
  if (!/^[-+]?\d*\.?\d+$/.test(cleaned)) return null
  return Number(cleaned)
}

function tableToCsv(table) {
  const escapeCell = (cell) => `"${String(cell).replace(/"/g, '""')}"`
  return [table.headers, ...table.rows].map((row) => row.map(escapeCell).join(',')).join('\n')
}

function chartFromTable(table) {
  const labelIndex = 0
  let valueIndex = -1
  for (let column = 1; column < table.headers.length; column += 1) {
    const values = table.rows.map((row) => numericValue(row[column])).filter((value) => value !== null)
    if (values.length >= Math.min(3, table.rows.length)) {
      valueIndex = column
      break
    }
  }
  if (valueIndex < 0) return null
  const points = table.rows
    .map((row) => ({
      label: row[labelIndex] || 'Item',
      value: numericValue(row[valueIndex]),
    }))
    .filter((point) => point.value !== null)
    .slice(0, 8)
  const max = Math.max(...points.map((point) => Math.abs(point.value)), 0)
  if (!points.length || max <= 0) return null
  return {
    title: `${table.headers[valueIndex]} by ${table.headers[labelIndex]}`,
    valueLabel: table.headers[valueIndex],
    points,
    max,
  }
}

function parseResearchSections(text = '') {
  const lines = text.split('\n')
  const sections = []
  let current = null
  const usefulHeading = /\b(summary|finding|analysis|source|citation|recommendation|pros|cons|risk|timeline|conclusion|report|overview|comparison)\b/i

  for (const line of lines) {
    const heading = line.match(/^(#{2,3})\s+(.+?)\s*$/)
    if (heading) {
      if (current?.content.trim()) sections.push(current)
      current = { title: heading[2].replace(/[*`]/g, '').trim(), content: '' }
    } else if (current) {
      current.content += `${line}\n`
    }
  }
  if (current?.content.trim()) sections.push(current)

  const hasSourceLinks = /\[[^\]]+\]\(https?:\/\/[^)]+\)/.test(text) || /^sources?:/im.test(text)
  const reportLike = sections.length >= 3 && (sections.some((section) => usefulHeading.test(section.title)) || hasSourceLinks)
  return reportLike ? sections.slice(0, 6) : []
}

const AnswerCards = memo(function AnswerCards({ message, sessionId }) {
  const text = message.text || ''
  const tables = parseMarkdownTables(text)
  const chart = tables.map(chartFromTable).find(Boolean)
  const reportSections = parseResearchSections(text)
  const pdfCitations = message.cards?.pdfCitations || []
  const document = message.cards?.document || null
  const researchPdfUrl = message.researchRunId && sessionId
    ? `${API_BASE}/api/chats/${encodeURIComponent(sessionId)}/research-runs/${encodeURIComponent(message.researchRunId)}/export.pdf`
    : ''
  const [activeReportTab, setActiveReportTab] = useState(0)
  const [copiedTable, setCopiedTable] = useState('')
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false)
  const [pdfOpenError, setPdfOpenError] = useState('')

  if (!tables.length && !chart && !reportSections.length && !pdfCitations.length && !researchPdfUrl) return null

  const copyTable = async (table, key) => {
    try {
      await navigator.clipboard.writeText(tableToCsv(table))
      setCopiedTable(key)
      window.setTimeout(() => setCopiedTable(''), 1400)
    } catch {
      setCopiedTable('')
    }
  }

  const downloadResearchPdf = async () => {
    if (!researchPdfUrl || isDownloadingPdf) return
    setIsDownloadingPdf(true)
    setPdfOpenError('')
    try {
      const response = await apiFetch(researchPdfUrl, { credentials: 'include' })
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.detail || 'Nexa could not create the research PDF.')
      }
      const pdfBytes = await response.arrayBuffer()
      const pdfBlob = new Blob([pdfBytes], { type: 'application/pdf' })
      const header = response.headers.get('content-disposition') || ''
      const filename = /filename="?([^";]+)"?/i.exec(header)?.[1] || 'nexa-research-report.pdf'
      const downloadUrl = URL.createObjectURL(pdfBlob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = filename.endsWith('.pdf') ? filename : `${filename}.pdf`
      link.style.display = 'none'
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0)
    } catch (error) {
      setPdfOpenError(error.message || 'Nexa could not download the research PDF.')
    } finally {
      setIsDownloadingPdf(false)
    }
  }

  return (
    <div className="answer-card-stack">
      {researchPdfUrl && (
        <section className="answer-card report-export-card">
          <div className="answer-card-head">
            <div>
              <span>RESEARCH EXPORT</span>
              <strong>Full PDF report</strong>
            </div>
            <button className="report-pdf-link" type="button" onClick={downloadResearchPdf} disabled={isDownloadingPdf}>
              <span className="report-pdf-mark" aria-hidden="true"><img src="/pdf.png" alt="" /></span>
              <span className="report-pdf-copy">
                <p>{isDownloadingPdf ? 'Preparing PDF...' : 'Download Pdf'}</p>
                <small>{isDownloadingPdf ? 'Creating document' : 'Save to your device'}</small>
              </span>
            </button>
          </div>
          {pdfOpenError && <p className="report-pdf-error" role="alert">{pdfOpenError}</p>}
        </section>
      )}

      {pdfCitations.length > 0 && (
        <section className="answer-card citation-card">
          <div className="answer-card-head">
            <div>
              <span>PDF SOURCES</span>
              <strong>{document?.filename || 'Uploaded PDF'}</strong>
            </div>
            {document?.page_count && <em>{document.page_count} pages</em>}
          </div>
          <div className="citation-grid">
            {pdfCitations.slice(0, 6).map((citation) => (
              <article className="citation-tile" key={`${citation.page}-${citation.chunk_index}`}>
                <span>Page {citation.page}</span>
                <p>{citation.preview}</p>
              </article>
            ))}
          </div>
        </section>
      )}

      {tables.length > 0 && (
        <section className="answer-card table-card">
          <div className="answer-card-head">
            <div>
              <span>TABLE VIEW</span>
              <strong>{tables.length === 1 ? 'Structured data' : `${tables.length} tables found`}</strong>
            </div>
          </div>
          {tables.map((table, tableIndex) => {
            const key = `table-${message.id}-${tableIndex}`
            return (
              <div className="interactive-table-wrap" key={key}>
                <div className="table-toolbar">
                  <span>{table.headers.join(' / ')}</span>
                  <button type="button" onClick={() => copyTable(table, key)}>
                    {copiedTable === key ? 'Copied' : 'Copy CSV'}
                  </button>
                </div>
                <div className="interactive-table-scroll">
                  <table>
                    <thead>
                      <tr>{table.headers.map((header) => <th key={header}>{header}</th>)}</tr>
                    </thead>
                    <tbody>
                      {table.rows.map((row, rowIndex) => (
                        <tr key={`${key}-row-${rowIndex}`}>
                          {row.map((cell, cellIndex) => <td key={`${key}-${rowIndex}-${cellIndex}`}>{cell}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })}
        </section>
      )}

      {chart && (
        <section className="answer-card chart-card">
          <div className="answer-card-head">
            <div>
              <span>CHART</span>
              <strong>{chart.title}</strong>
            </div>
            <em>{chart.valueLabel}</em>
          </div>
          <div className="bar-chart" role="img" aria-label={chart.title}>
            {chart.points.map((point) => (
              <div className="bar-row" key={`${point.label}-${point.value}`}>
                <span>{point.label}</span>
                <div><i style={{ width: `${Math.max(5, (Math.abs(point.value) / chart.max) * 100)}%` }} /></div>
                <strong>{point.value}</strong>
              </div>
            ))}
          </div>
        </section>
      )}

      {reportSections.length > 0 && (
        <section className="answer-card report-card">
          <div className="answer-card-head">
            <div>
              <span>REPORT VIEW</span>
              <strong>Section navigator</strong>
            </div>
          </div>
          <div className="report-tabs" role="tablist" aria-label="Report sections">
            {reportSections.map((section, index) => (
              <button
                key={section.title}
                type="button"
                className={activeReportTab === index ? 'active' : ''}
                onClick={() => setActiveReportTab(index)}
              >
                {section.title}
              </button>
            ))}
          </div>
          <div className="report-panel">
            <MarkdownResponse>{reportSections[activeReportTab]?.content || ''}</MarkdownResponse>
          </div>
        </section>
      )}
    </div>
  )
})

const AssistantResponseActions = memo(function AssistantResponseActions({ message, canSaveFeedback, onFeedback }) {
  const [copied, setCopied] = useState(false)
  const [busy, setBusy] = useState(false)

  const copyResponse = async () => {
    try {
      await navigator.clipboard.writeText(String(message.text || ''))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      setCopied(false)
    }
  }

  const submitFeedback = async (reaction) => {
    if (!canSaveFeedback || busy || message.feedback === reaction) return
    setBusy(true)
    try {
      await onFeedback(message, reaction)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="assistant-response-actions" aria-label="Response actions">
      <button
        type="button"
        className={message.feedback === 'like' ? 'is-selected' : ''}
        onClick={() => submitFeedback('like')}
        disabled={!canSaveFeedback || busy}
        aria-label="Like response"
        title="Like"
      ><ThumbsUp size={14} /></button>
      <button
        type="button"
        className={message.feedback === 'dislike' ? 'is-selected' : ''}
        onClick={() => submitFeedback('dislike')}
        disabled={!canSaveFeedback || busy}
        aria-label="Dislike response"
        title="Dislike"
      ><ThumbsDown size={14} /></button>
      {/* <span aria-hidden="true" /> */}
      <button type="button" onClick={copyResponse} aria-label="Copy response" title={copied ? 'Copied' : 'Copy response'}>
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </div>
  )
})

function Logo() {
  return <img className="logo-mark" src="/NEXA.png" alt="" aria-hidden="true" />
}

const initialiseParticles = async (engine) => {
  await loadSlim(engine)
}

function AmbientParticles({ id = 'nexa-particles', className = 'ambient-particles', compact = false }) {
  const palette = compact
    ? ['#8f7bff', '#9fb5ff']
    : ['#8f7bff', '#8aa4ff', '#f4f1ff']
  return <ParticlesProvider init={initialiseParticles}><Particles
    id={id}
    className={className}
    options={{
      background: { color: { value: 'transparent' } },
      fpsLimit: 50,
      detectRetina: true,
      interactivity: {
        events: {
          onHover: { enable: !compact, mode: 'bubble' },
          resize: { enable: true },
        },
        modes: {
          bubble: {
            distance: 140,
            duration: 1.8,
            opacity: 0.55,
            size: 4,
          },
        },
      },
      particles: {
        color: { value: palette },
        links: { enable: false },
        move: {
          enable: true,
          speed: compact ? 0.22 : 0.34,
          direction: 'top-right',
          random: true,
          straight: false,
          outModes: { default: 'out' },
        },
        number: {
          density: { enable: !compact, width: 1100, height: 700 },
          value: compact ? 12 : 42,
        },
        opacity: {
          value: { min: compact ? 0.08 : 0.1, max: compact ? 0.28 : 0.48 },
          animation: {
            enable: true,
            speed: compact ? 0.35 : 0.55,
            sync: false,
          },
        },
        shadow: {
          enable: true,
          color: '#8f7bff',
          blur: compact ? 5 : 9,
        },
        shape: { type: 'circle' },
        size: {
          value: { min: compact ? 0.8 : 0.9, max: compact ? 2.2 : 3.8 },
          animation: {
            enable: true,
            speed: compact ? 0.6 : 0.85,
            sync: false,
          },
        },
      },
    }}
  /></ParticlesProvider>
}

function AuthScreen({ onSignedIn }) {
  const [mode, setMode] = useState('login')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const response = await apiFetch(`${API_BASE}/api/auth/${mode === 'login' ? 'login' : 'register'}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
        body: JSON.stringify({ name, email, password }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not sign in.')
      writeAuthToken(data.token || '')
      onSignedIn(data.user)
    } catch (requestError) {
      setError(requestError.message || 'Could not sign in.')
    } finally { setBusy(false) }
  }

  const startGoogleSignIn = () => {
    writePendingInviteToken(inviteTokenFromPath())
    window.location.assign(`${API_BASE}/api/auth/google`)
  }

  return <main className="auth-screen"><AmbientParticles /><section className="auth-card">
    <div className="auth-brand"><Logo /><span>NEXA</span></div>
    <p className="section-label">PRIVATE WORKSPACE</p>
    <h1>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h1>
    <p>Sign in to save and revisit your chat sessions.</p>
    <button className="google-signin" type="button" onClick={startGoogleSignIn}>
      <span><img src="/google.png" alt="Google" style = {{ width: '20px', height: '20px' }} /></span> Sign in with Google
    </button>
    <div className="auth-divider"><span>or</span></div>
    <form onSubmit={submit} className="auth-form">
      {mode === 'register' && <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" autoComplete="name" />}
      <input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Email address" type="email" autoComplete="email" required />
      <input value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" type="password" minLength="8" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required />
      {error && <p className="auth-error">{error}</p>}
      <button className="auth-submit" disabled={busy}>{busy ? 'Please wait...' : mode === 'login' ? 'Sign in' : 'Create account'}</button>
    </form>
    <button type="button" className="auth-switch" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError('') }}>
      {mode === 'login' ? 'New here? Create an account' : 'Already have an account? Sign in'}
    </button>
  </section></main>
}

function App() {
  const [messages, setMessages] = useState(starterMessages)
  const [input, setInput] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [apiError, setApiError] = useState('')
  const [isOnline, setIsOnline] = useState(false)
  const [thinkingStatus, setThinkingStatus] = useState('')
  const [thinkingDetail, setThinkingDetail] = useState('')
  const [thinkingEvents, setThinkingEvents] = useState([])
  const [liveAnswer, setLiveAnswer] = useState('')
  const [mcpServers, setMcpServers] = useState([])
  const [googleServices, setGoogleServices] = useState(() => mergeGoogleServices())
  const [capabilities, setCapabilities] = useState({ local: [], google: [] })
  const [googleActionBusy, setGoogleActionBusy] = useState('')
  const [pendingEmail, setPendingEmail] = useState(null)
  const [pendingMcpAction, setPendingMcpAction] = useState(null)
  const [emailActionBusy, setEmailActionBusy] = useState('')
  const [mcpActionBusy, setMcpActionBusy] = useState('')
  const [pendingRecipient, setPendingRecipient] = useState('')
  const [pendingCc, setPendingCc] = useState('')
  const [pendingBcc, setPendingBcc] = useState('')
  const [pdfFile, setPdfFile] = useState(null)
  const [user, setUser] = useState(null)
  const [authView, setAuthView] = useState(false)
  const [chatSessions, setChatSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [deletingSessionId, setDeletingSessionId] = useState('')
  const [participants, setParticipants] = useState([])
  const [typingUsers, setTypingUsers] = useState([])
  const [sessionRole, setSessionRole] = useState('member')
  const [membersOpen, setMembersOpen] = useState(false)
  const [memberBusy, setMemberBusy] = useState('')
  const [inviteLink, setInviteLink] = useState('')
  const [shareHistoryMode, setShareHistoryMode] = useState('none')
  const [shareDialogOpen, setShareDialogOpen] = useState(false)
  const [inviteCopied, setInviteCopied] = useState(false)
  const [joinedInviteSessionId, setJoinedInviteSessionId] = useState('')
  const [pendingRejoinInvite, setPendingRejoinInvite] = useState(null)
  const [rejoinBusy, setRejoinBusy] = useState('')
  const [replyTarget, setReplyTarget] = useState(null)
  const [composerHelpOpen, setComposerHelpOpen] = useState(false)
  const [visibleMessageDay, setVisibleMessageDay] = useState('')
  const [isConversationScrolled, setIsConversationScrolled] = useState(false)
  const [browserLocation, setBrowserLocation] = useState(() => ({
    status: 'idle',
    location: null,
    error: '',
    updatedAt: 0,
  }))
  const [leftPanelOpen, setLeftPanelOpen] = useState(false)
  const [theme, setTheme] = useState(readThemePreference)
  const feedRef = useRef(null)
  const shouldStickToBottomRef = useRef(true)
  const scrollFrameRef = useRef(0)
  const composerInputRef = useRef(null)
  const sessionSocketRef = useRef(null)
  const typingStopTimerRef = useRef(0)
  const typingActiveRef = useRef(false)
  const membersPanelRef = useRef(null)
  const composerHelpRef = useRef(null)
  const pdfInputRef = useRef(null)
  const locationPromptedRef = useRef('')
  const shareInviteRequestRef = useRef(0)

  const applyPendingEmail = useCallback((email) => {
    setPendingEmail(email || null)
    setPendingRecipient(email?.to?.join(', ') || '')
    setPendingCc(email?.cc?.join(', ') || '')
    setPendingBcc(email?.bcc?.join(', ') || '')
  }, [])

  const loadChatSessions = useCallback(async () => {
    const response = await apiFetch(`${API_BASE}/api/chats`, { credentials: 'include' })
    if (!response.ok) throw new Error('Could not load chat sessions.')
    const sessions = (await response.json()).sessions || []
    setChatSessions((current) => areSessionsEquivalent(current, sessions) ? current : sessions)
    return sessions
  }, [])

  const createChatSession = useCallback(async () => {
    const response = await apiFetch(`${API_BASE}/api/chats`, { method: 'POST', credentials: 'include' })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || 'Could not create a chat.')
    setChatSessions((current) => [data.session, ...current])
    setActiveSessionId(data.session.id)
    setMessages(starterMessages)
    return data.session
  }, [])

  const markSessionRead = useCallback(async (sessionId) => {
    const response = await apiFetch(`${API_BASE}/api/chats/${sessionId}/read`, {
      method: 'POST',
      credentials: 'include',
    })
    if (!response.ok) return
    setChatSessions((current) => current.map((session) => (
      session.id === sessionId ? { ...session, unread_count: 0 } : session
    )))
  }, [])

  const openChatSession = useCallback(async (sessionId, options = {}) => {
    const response = await apiFetch(`${API_BASE}/api/chats/${sessionId}/messages`, { credentials: 'include' })
    const data = await response.json()
    if (!response.ok) {
      const error = new Error(data.detail || 'Could not open this chat.')
      error.status = response.status
      throw error
    }
    setActiveSessionId(sessionId)
    setReplyTarget(null)
    const nextMessages = data.messages.length
      ? data.messages.map((message, index) => messageWithDocumentReference(message, message.id || `${sessionId}-${index}`))
      : starterMessages
    setMessages((current) => areMessagesEquivalent(current, nextMessages) ? current : nextMessages)
    if (options.markRead !== false) markSessionRead(sessionId).catch(() => {})
  }, [markSessionRead])

  const loadParticipants = useCallback(async (sessionId = activeSessionId) => {
    if (!sessionId) return
    const response = await apiFetch(`${API_BASE}/api/chats/${sessionId}/participants`, { credentials: 'include' })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || 'Could not load chat members.')
    setParticipants(data.participants || [])
    setSessionRole(data.your_role || 'member')
  }, [activeSessionId])

  const createShareInvite = useCallback(async (historyMode = 'none', options = {}) => {
    if (!activeSessionId || sessionRole !== 'admin') return
    const requestId = shareInviteRequestRef.current + 1
    shareInviteRequestRef.current = requestId
    const mode = historyMode || 'none'
    if (options.resetDialog) {
      setShareHistoryMode(mode)
      setInviteLink('')
      setInviteCopied(false)
      setShareDialogOpen(true)
    }
    setMemberBusy('invite')
    try {
      const response = await apiFetch(`${API_BASE}/api/chats/${activeSessionId}/invites`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ history_mode: mode }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not create an invite.')
      if (requestId !== shareInviteRequestRef.current) return
      const link = `${window.location.origin}/join/${data.token}`
      setInviteLink(link)
      setShareHistoryMode(mode)
      setInviteCopied(false)
      setShareDialogOpen(true)
    } catch (error) {
      if (requestId !== shareInviteRequestRef.current) return
      setApiError(error.message || 'Could not copy the invite link.')
    } finally {
      if (requestId === shareInviteRequestRef.current) setMemberBusy('')
    }
  }, [activeSessionId, sessionRole])

  const shareChat = useCallback(() => {
    createShareInvite('none', { resetDialog: true })
  }, [createShareInvite])

  const chooseShareHistoryMode = useCallback((mode) => {
    const nextMode = mode || 'none'
    setShareHistoryMode(nextMode)
    setInviteLink('')
    setInviteCopied(false)
    createShareInvite(nextMode)
  }, [createShareInvite])

  const closeShareDialog = useCallback(() => {
    shareInviteRequestRef.current += 1
    setShareDialogOpen(false)
    setShareHistoryMode('none')
    setInviteLink('')
    setInviteCopied(false)
    setMemberBusy('')
  }, [])

  const copyInviteLink = useCallback(async () => {
    if (!inviteLink) return
    try {
      await navigator.clipboard.writeText(inviteLink)
      setInviteCopied(true)
    } catch {
      setApiError('Could not copy the invite link.')
    }
  }, [inviteLink])

  const changeParticipantRole = useCallback(async (participant, role) => {
    setMemberBusy(participant.user_id)
    try {
      const response = await apiFetch(`${API_BASE}/api/chats/${activeSessionId}/participants/${participant.user_id}/role`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify({ role }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not update this member.')
      await loadParticipants()
      await loadChatSessions()
    } catch (error) {
      setApiError(error.message)
    } finally {
      setMemberBusy('')
    }
  }, [activeSessionId, loadChatSessions, loadParticipants])

  const removeParticipant = useCallback(async (participant) => {
    setMemberBusy(participant.user_id)
    try {
      const response = await apiFetch(`${API_BASE}/api/chats/${activeSessionId}/participants/${participant.user_id}`, { method: 'DELETE', credentials: 'include' })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not remove this member.')
      await loadParticipants()
      await loadChatSessions()
    } catch (error) {
      setApiError(error.message)
    } finally {
      setMemberBusy('')
    }
  }, [activeSessionId, loadChatSessions, loadParticipants])

  const deleteChatSession = useCallback(async (sessionId) => {
    if (!sessionId || deletingSessionId) return
    setDeletingSessionId(sessionId)
    setApiError('')
    try {
      const response = await apiFetch(`${API_BASE}/api/chats/${sessionId}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not delete this chat.')

      const remainingSessions = data.private_session
        ? [data.private_session, ...chatSessions.filter((session) => session.id !== sessionId && session.id !== data.private_session.id)]
        : chatSessions.filter((session) => session.id !== sessionId)
      setChatSessions(remainingSessions)
      if (sessionId === activeSessionId) {
        if (data.private_session) await openChatSession(data.private_session.id)
        else if (remainingSessions.length) await openChatSession(remainingSessions[0].id)
        else await createChatSession()
      }
    } catch (error) {
      setApiError(error.message === 'Failed to fetch'
        ? 'Nexa API is offline. Start the backend and try again.'
        : error.message)
    } finally {
      setDeletingSessionId('')
    }
  }, [activeSessionId, chatSessions, createChatSession, deletingSessionId, openChatSession])

  const activeSessionHasMessages = messages.some((message) => (
    message.role !== 'system' && String(message.id) !== '1'
  ))
  const activeSessionIsEmpty = Boolean(activeSessionId) && !activeSessionHasMessages
  const activeSessionIsShared = Boolean(chatSessions.find((session) => session.id === activeSessionId)?.shared)
  const openOrCreateDraftSession = useCallback(async () => {
    if (!user) {
      setAuthView(true)
      return
    }
    setApiError('')
    setJoinedInviteSessionId('')
    const existingDraft = chatSessions.find((session) => (
      session.id !== activeSessionId
      && !session.shared
      && session.title === 'New chat'
    ))
    if (existingDraft) {
      await openChatSession(existingDraft.id)
      setLeftPanelOpen(false)
      return
    }
    await createChatSession()
    setLeftPanelOpen(false)
  }, [activeSessionId, chatSessions, createChatSession, openChatSession, user])
  const isAssistantLikeMessage = useCallback((message) => (
    message?.role === 'assistant' || Boolean(message?.researchRunId)
  ), [])
  const messageAuthorLabel = useCallback((message) => (
    isAssistantLikeMessage(message)
      ? 'Nexa'
      : (message?.senderName || (message?.sender_user_id === user?.id ? 'You' : 'Member'))
  ), [isAssistantLikeMessage, user?.id])
  const systemMessageText = useCallback((message) => {
    if (!message?.systemAction) return message?.text || ''
    const isTarget = message.target_user_id && message.target_user_id === user?.id
    const subject = isTarget ? 'You' : (message.targetEmail || 'A user')
    if (message.systemAction === 'added') return `${subject} ${isTarget ? 'were' : 'is'} added`
    if (message.systemAction === 'removed') return `${subject} ${isTarget ? 'were' : 'is'} removed`
    if (message.systemAction === 'left') return `${subject} left the chat`
    return message.text || ''
  }, [user?.id])
  const fallbackMessageDay = messages.find((message) => message.role !== 'system' && message.createdAt)?.createdAt
  const sessionDayLabel = visibleMessageDay || (fallbackMessageDay ? formatMessageDay(fallbackMessageDay) : '')
  const typingLabel = typingUsers.length === 1
    ? `${typingUsers[0].name} is typing...`
    : typingUsers.length > 1
      ? `${typingUsers.slice(0, 2).map((person) => person.name).join(' and ')} are typing...`
      : ''

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      window.localStorage.setItem(THEME_KEY, theme)
    } catch {
      // Theme preference persistence is optional.
    }
  }, [theme])

  useEffect(() => {
    const feed = feedRef.current
    if (!feed) return undefined

    let updateFrame = 0
    const updateVisibleDay = () => {
      updateFrame = 0
      const messageNodes = Array.from(feed.querySelectorAll('[data-message-day]'))
      if (!messageNodes.length) {
        setVisibleMessageDay((current) => current ? '' : current)
        return
      }
      const feedTop = feed.getBoundingClientRect().top
      const markerLine = feedTop + 18
      const visibleNode = messageNodes.find((node) => node.getBoundingClientRect().bottom >= markerLine) || messageNodes[messageNodes.length - 1]
      const nextDay = visibleNode?.dataset.messageDay || ''
      setVisibleMessageDay((current) => current === nextDay ? current : nextDay)
    }

    const scheduleVisibleDayUpdate = () => {
      const distanceFromBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight
      shouldStickToBottomRef.current = distanceFromBottom < 96
      setIsConversationScrolled((current) => {
        const next = feed.scrollTop > 8
        return current === next ? current : next
      })
      if (!updateFrame) updateFrame = window.requestAnimationFrame(updateVisibleDay)
    }

    updateVisibleDay()
    scheduleVisibleDayUpdate()
    feed.addEventListener('scroll', scheduleVisibleDayUpdate, { passive: true })
    window.addEventListener('resize', scheduleVisibleDayUpdate)
    return () => {
      feed.removeEventListener('scroll', scheduleVisibleDayUpdate)
      window.removeEventListener('resize', scheduleVisibleDayUpdate)
      if (updateFrame) window.cancelAnimationFrame(updateFrame)
    }
  }, [activeSessionId, messages.length])

  const startReply = useCallback((message) => {
    const preview = replyPreviewFromMessage(message)
    if (!preview) return
    setReplyTarget(preview)
  }, [])

  const submitAssistantFeedback = useCallback(async (message, reaction) => {
    if (!activeSessionId || !message?.id) return
    try {
      const response = await apiFetch(
        `${API_BASE}/api/chats/${encodeURIComponent(activeSessionId)}/messages/${encodeURIComponent(message.id)}/feedback`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ reaction }),
        },
      )
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail || 'Could not save feedback.')
      setMessages((current) => current.map((item) => (
        String(item.id) === String(message.id) ? { ...item, feedback: data.feedback || reaction } : item
      )))
    } catch (error) {
      setApiError(error.message || 'Could not save feedback.')
    }
  }, [activeSessionId])

  const activateAgentMention = useCallback(() => {
    setInput((current) => {
      const remainder = stripAgentDraftMention(current)
      return remainder ? `@Nexa ${remainder}` : '@Nexa '
    })
    window.requestAnimationFrame(() => composerInputRef.current?.focus())
  }, [])

  const clearAgentMention = useCallback(() => {
    setInput((current) => stripAgentMention(current))
    window.requestAnimationFrame(() => composerInputRef.current?.focus())
  }, [])

  const insertComposerCommand = useCallback((command) => {
    setInput((current) => {
      const trimmed = current.trimStart()
      if (!current.trim()) return command
      if (trimmed.toLowerCase().startsWith(command.toLowerCase())) return current
      return `${command}${current.trimStart()}`
    })
    setComposerHelpOpen(false)
  }, [])

  const handleTypingPresence = useCallback((isTyping) => {
    if (!activeSessionIsShared) {
      typingActiveRef.current = false
      return
    }
    const socket = sessionSocketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) return
    if (typingActiveRef.current === isTyping) return
    typingActiveRef.current = isTyping
    try {
      socket.send(JSON.stringify({ type: 'typing', is_typing: isTyping }))
    } catch {
      typingActiveRef.current = false
    }
  }, [activeSessionIsShared])

  const announceTyping = useCallback((isTyping) => {
    window.clearTimeout(typingStopTimerRef.current)
    handleTypingPresence(isTyping)
    if (isTyping) {
      typingStopTimerRef.current = window.setTimeout(() => {
        handleTypingPresence(false)
      }, 1200)
    }
  }, [handleTypingPresence])

  const requestSignedInLocation = useCallback(async (signedInUser, { force = false } = {}) => {
    if (!signedInUser) return null
    const key = userLocationCacheKey(signedInUser)
    if (!key) return null
    const cached = readCachedLocation(signedInUser)
    if (!force && cached?.status === 'denied') {
      setBrowserLocation({
        status: 'denied',
        location: null,
        error: cached.error || '',
        updatedAt: cached.updatedAt || 0,
      })
      return null
    }
    if (!force && cached?.status === 'granted' && cached.location && !cached.stale) {
      setBrowserLocation(cached)
      return cached.location
    }

    setBrowserLocation((current) => ({ ...current, status: 'requesting', error: '' }))
    const result = await currentBrowserLocation()
    const next = result.location
      ? {
        status: 'granted',
        location: result.location,
        error: '',
        updatedAt: Date.now(),
      }
      : {
        status: 'denied',
        location: null,
        error: result.error || 'Nexa could not access browser location.',
        updatedAt: Date.now(),
      }
    setBrowserLocation(next)
    writeCachedLocation(signedInUser, next)
    return next.location
  }, [])

  const acceptInviteFromUrl = useCallback(async (options = {}) => {
    const token = inviteTokenFromPath() || readPendingInviteToken()
    if (!token) return false
    writePendingInviteToken(token)
    const body = { token }
    if (typeof options.sharePrivateConversation === 'boolean') {
      body.share_private_conversation = options.sharePrivateConversation
    }
    const response = await apiFetch(`${API_BASE}/api/chats/invites/accept`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(body),
    })
    const data = await response.json()
    if (!response.ok) {
      if (response.status === 409 && data.detail?.code === 'rejoin_private_copy_choice_required') {
        setPendingRejoinInvite({ token, sessionTitle: data.detail.session_title || 'this session' })
        setAuthView(false)
        return true
      }
      writePendingInviteToken('')
      throw new Error(data.detail || 'Could not join this chat.')
    }
    writePendingInviteToken('')
    setPendingRejoinInvite(null)
    window.history.replaceState({}, '', '/')
    setJoinedInviteSessionId(data.session.invite_history_mode === 'none' && !data.session.rejoined ? data.session.id : '')
    setChatSessions((current) => {
      if (!data.session) return current
      const withoutJoined = current.filter((session) => session.id !== data.session.id)
      return [data.session, ...withoutJoined]
    })
    await loadChatSessions()
    await openChatSession(data.session.id)
    return true
  }, [loadChatSessions, openChatSession])

  const finishRejoinInvite = useCallback(async (sharePrivateConversation) => {
    if (!pendingRejoinInvite?.token || rejoinBusy) return
    setRejoinBusy(sharePrivateConversation ? 'share' : 'private')
    try {
      writePendingInviteToken(pendingRejoinInvite.token)
      await acceptInviteFromUrl({ sharePrivateConversation })
    } catch (error) {
      setApiError(error.message || 'Could not rejoin this session.')
    } finally {
      setRejoinBusy('')
    }
  }, [acceptInviteFromUrl, pendingRejoinInvite, rejoinBusy])

  const completeSignIn = useCallback(async (signedInUser) => {
    setUser(signedInUser)
    setAuthView(false)
    try {
      if (await acceptInviteFromUrl()) return
      const sessions = await loadChatSessions()
      if (sessions.length) await openChatSession(sessions[0].id)
      else await createChatSession()
    } catch (error) { setApiError(error.message) }
  }, [acceptInviteFromUrl, createChatSession, loadChatSessions, openChatSession])

  const loadPendingEmail = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/email/pending`, { credentials: 'include' })
      if (!response.ok) return
      const data = await response.json()
      applyPendingEmail(data.pending_email || null)
    } catch {
      // This card is secondary UI state and can fail quietly.
    }
  }, [applyPendingEmail])

  const loadMcpServers = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/mcp/servers`, { credentials: 'include' })
      if (!response.ok) return
      const data = await response.json()
      setMcpServers(data.servers || [])
    } catch {
      // Non-critical connected service status.
    }
  }, [])

  const loadGoogleServices = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/google/status`, { credentials: 'include' })
      if (!response.ok) return
      const data = await response.json()
      setGoogleServices(mergeGoogleServices(data.services || []))
      writePendingGoogleService('')
    } catch {
      // Google connections are optional and should not affect the chat UI.
    }
  }, [])

  const loadCapabilities = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/capabilities`, { credentials: 'include' })
      if (!response.ok) return
      setCapabilities(await response.json())
    } catch {
      // Capability status is informational and can fail quietly.
    }
  }, [])

  const connectGoogleService = useCallback((service) => {
    if (googleActionBusy) return
    setGoogleActionBusy(service)
    writePendingGoogleService(service)
    setGoogleServices((current) => mergeGoogleServices(current).map((item) => (
      item.service === service ? { ...item, connecting: true } : item
    )))
    const url = new URL(`${API_BASE}/api/google/connect/${service}`, window.location.origin)
    const token = readAuthToken()
    if (token) url.searchParams.set('auth_token', token)
    window.location.assign(url.toString())
  }, [googleActionBusy])

  const disconnectGoogleService = useCallback(async (service) => {
    if (googleActionBusy) return
    setGoogleActionBusy(service)
    setApiError('')
    try {
      const response = await apiFetch(`${API_BASE}/api/google/disconnect/${service}`, {
        method: 'POST',
        credentials: 'include',
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not disconnect Google service.')
      setGoogleServices((current) => mergeGoogleServices(current).map((item) => (
        item.service === service ? { ...item, connected: false, email: '' } : item
      )))
      await loadGoogleServices()
      await loadMcpServers()
      await loadCapabilities()
    } catch (error) {
      setApiError(error.message || 'Could not disconnect Google service.')
    } finally {
      setGoogleActionBusy('')
    }
  }, [googleActionBusy, loadGoogleServices, loadMcpServers, loadCapabilities])

  const loadPendingMcpAction = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/mcp/pending`, { credentials: 'include' })
      if (!response.ok) return
      const data = await response.json()
      setPendingMcpAction(data.pending_action || null)
    } catch {
      // Secondary UI state.
    }
  }, [])

  const clearPdfAttachment = useCallback(() => {
    setPdfFile(null)
    if (pdfInputRef.current) pdfInputRef.current.value = ''
  }, [])

  const handlePdfChange = useCallback((event) => {
    const selectedFile = event.target.files?.[0] || null
    setApiError('')
    if (!selectedFile) {
      setPdfFile(null)
      return
    }
    if (selectedFile.type && selectedFile.type !== 'application/pdf') {
      setApiError('Upload a PDF file.')
      clearPdfAttachment()
      return
    }
    if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setApiError('Upload a PDF file.')
      clearPdfAttachment()
      return
    }
    if (selectedFile.size > MAX_DOCUMENT_BYTES) {
      setApiError('Please upload files less than 5 MB.')
      clearPdfAttachment()
      return
    }
    setPdfFile(selectedFile)
  }, [clearPdfAttachment])

  const handlePendingEmailAction = useCallback(async (action) => {
    if (!pendingEmail?.id || emailActionBusy) return
    if (action === 'confirm' && !pendingRecipient.trim()) {
      setApiError('Enter the recipient email address before sending.')
      return
    }
    setApiError('')
    setEmailActionBusy(action)

    try {
      const response = await apiFetch(
        `${API_BASE}/api/email/pending/${pendingEmail.id}/${action}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({
            recipient: pendingRecipient,
            cc: pendingCc,
            bcc: pendingBcc,
          }),
        },
      )
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || `Could not ${action} the email.`)

      applyPendingEmail(null)
      const createdAt = new Date().toISOString()
      setMessages((current) => [...current, {
        id: Date.now() + 2,
        role: 'assistant',
        text: data.message,
        time: formatMessageTime(createdAt),
        createdAt,
      }])
      setIsOnline(true)
    } catch (error) {
      setApiError(error.message === 'Failed to fetch'
        ? 'Nexa API is offline. Start the backend and try again.'
        : error.message)
    } finally {
      setEmailActionBusy('')
      loadPendingEmail()
    }
  }, [
    pendingEmail,
    emailActionBusy,
    loadPendingEmail,
    pendingRecipient,
    pendingCc,
    pendingBcc,
    applyPendingEmail,
  ])

  const handlePendingMcpAction = useCallback(async (action) => {
    if (!pendingMcpAction?.id || mcpActionBusy) return
    setApiError('')
    setMcpActionBusy(action)

    try {
      const response = await apiFetch(
        `${API_BASE}/api/mcp/pending/${pendingMcpAction.id}/${action}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
        },
      )
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || `Could not ${action} the connected action.`)

      setPendingMcpAction(null)
      const createdAt = new Date().toISOString()
      setMessages((current) => [...current, {
        id: Date.now() + 3,
        role: 'assistant',
        text: data.message,
        time: formatMessageTime(createdAt),
        createdAt,
      }])
      setIsOnline(true)
    } catch (error) {
      setApiError(error.message === 'Failed to fetch'
        ? 'Nexa API is offline. Start the backend and try again.'
        : error.message)
    } finally {
      setMcpActionBusy('')
      loadPendingMcpAction()
    }
  }, [pendingMcpAction, mcpActionBusy, loadPendingMcpAction])

  const sendMessage = useCallback(async (value = input) => {
    const cleanMessage = value.trim()
    if (!cleanMessage || isThinking) return
    if (!user) {
      setAuthView(true)
      return
    }
    if (!activeSessionId) {
      setApiError('Preparing a new chat session. Please try again.')
      return
    }

    const attachedPdf = pdfFile
    const selectedReply = replyTarget
    const sharedSession = Boolean(chatSessions.find((session) => session.id === activeSessionId)?.shared)
    const agentMatch = /^@nexa\b[\s,:-]*(.*)$/is.exec(cleanMessage)
    const usesAgent = !sharedSession || Boolean(agentMatch)
    const agentPrompt = agentMatch ? agentMatch[1].trim() : cleanMessage
    const researchMode = RESEARCH_COMMAND_PATTERN.test(agentPrompt)
    if (agentMatch && !agentPrompt) {
      setApiError('Add a prompt after @Nexa.')
      return
    }
    if (researchMode && !/^\/research\s+\S/i.test(agentPrompt)) {
      setApiError('Write a research question after /research.')
      return
    }
    if ((attachedPdf || /^\/doc(?:\s|$)/i.test(agentPrompt)) && !usesAgent) {
      setApiError('Start AI and document requests with @Nexa in shared sessions.')
      return
    }
    if (attachedPdf && researchMode) {
      setApiError('Deep research currently uses online sources. Remove the PDF or ask about it in a normal message.')
      return
    }
    const createdAt = new Date().toISOString()
    setMessages((current) => [
      ...current,
      {
        id: Date.now(),
        role: 'user',
        text: usesAgent ? agentPrompt : cleanMessage,
        ...(attachedPdf ? { documentName: attachedPdf.name } : {}),
        ...(selectedReply ? { replyTo: selectedReply } : {}),
        time: formatMessageTime(createdAt),
        createdAt,
      },
    ])
    if (attachedPdf) clearPdfAttachment()
    setInput('')
    setReplyTarget(null)
    setApiError('')
    announceTyping(false)
    if (!usesAgent) {
      try {
        const socket = sessionSocketRef.current
        if (socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({
            type: 'message',
            content: cleanMessage,
            ...(selectedReply ? { reply_to_id: selectedReply.id } : {}),
          }))
          setIsOnline(true)
          return
        }
        const response = await apiFetch(`${API_BASE}/api/chats/${activeSessionId}/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ message: cleanMessage, ...(selectedReply ? { reply_to_id: selectedReply.id } : {}) }),
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || 'Could not send the message.')
        if (data.message) {
          const incoming = messageWithDocumentReference(data.message, data.message.id)
          setMessages((current) => {
            const optimisticIndex = [...current].reverse().findIndex((message) => (
              message.role === 'user' && message.text === incoming.text && !message.senderName && !message.sender_user_id
            ))
            if (optimisticIndex < 0) {
              if (current.some((message) => String(message.id) === String(incoming.id))) return current
              return [...current, incoming]
            }
            const index = current.length - optimisticIndex - 1
            return current.map((message, itemIndex) => itemIndex === index ? incoming : message)
          })
        }
        loadChatSessions().catch(() => {})
        setIsOnline(true)
      } catch (error) {
        setApiError(error.message === 'Failed to fetch' ? 'Nexa API is offline. Start the backend and try again.' : error.message)
        setIsOnline(false)
      }
      return
    }
    setThinkingStatus(attachedPdf ? 'Reading attached PDF' : researchMode ? 'Preparing deep research' : 'Connecting to Nexa')
    setThinkingDetail(attachedPdf
      ? 'Extracting text, indexing chunks, and searching the document.'
      : researchMode
        ? 'Selecting only relevant verified research sources for this question.'
        : 'Opening a live stream to the local backend.')
    setThinkingEvents([
      {
        id: `thinking-${Date.now()}`,
        stage: attachedPdf ? 'PDF' : researchMode ? 'Research' : 'Connect',
        message: attachedPdf ? 'Reading attached PDF' : researchMode ? 'Preparing deep research' : 'Connecting to Nexa',
        detail: attachedPdf
          ? 'Extracting text, indexing chunks, and searching the document.'
          : researchMode
            ? 'Selecting only relevant verified research sources for this question.'
            : 'Opening a live stream to the local backend.',
      },
    ])
    setLiveAnswer('')
    setIsThinking(true)

    try {
      if (attachedPdf) {
        const formData = new FormData()
        formData.append('question', agentPrompt)
        formData.append('session_id', activeSessionId)
        formData.append('file', attachedPdf)
        const response = await apiFetch(`${API_BASE}/api/pdf/ask`, {
          method: 'POST',
          body: formData,
          credentials: 'include',
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || 'Nexa could not answer from that PDF.')

        const citations = data.citations || []
        const citationLabels = [...new Set(citations.map((citation) => (
          citation.filename ? `[${citation.filename} · page ${citation.page}]` : `[page ${citation.page}]`
        )).filter((label) => !label.includes('page undefined')))]
        const citationNote = citationLabels.length
          ? `\n\nSources: ${citationLabels.join(', ')}`
          : ''
        const completedAnswer = `${data.answer || ''}${citationNote}`.trim()
        if (!completedAnswer) throw new Error('Nexa returned an empty PDF answer.')
        const answerCreatedAt = new Date().toISOString()
        setMessages((current) => [...current, {
          id: Date.now() + 1,
          role: 'assistant',
          text: completedAnswer,
          time: formatMessageTime(answerCreatedAt),
          createdAt: answerCreatedAt,
          cards: {
            document: data.document || null,
            pdfCitations: data.citations || [],
          },
        }])
        clearPdfAttachment()
        loadChatSessions().catch(() => {})
        setIsOnline(true)
        return
      }

      if (/^\/doc(?:\s|$)/i.test(agentPrompt)) {
        const response = await apiFetch(`${API_BASE}/api/documents/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ question: agentPrompt, session_id: activeSessionId }),
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || 'Nexa could not search your saved documents.')
        const citations = data.citations || []
        const citationLabels = [...new Set(citations.map((citation) => (
          citation.filename ? `[${citation.filename} · page ${citation.page}]` : `[page ${citation.page}]`
        )).filter((label) => !label.includes('page undefined')))]
        const citationNote = citationLabels.length ? `\n\nSources: ${citationLabels.join(', ')}` : ''
        const completedAnswer = `${data.answer || ''}${citationNote}`.trim()
        if (!completedAnswer) throw new Error('Nexa returned an empty document answer.')
        const answerCreatedAt = new Date().toISOString()
        setMessages((current) => [...current, {
          id: Date.now() + 1,
          role: 'assistant',
          text: completedAnswer,
          time: formatMessageTime(answerCreatedAt),
          createdAt: answerCreatedAt,
          cards: { document: data.document || null, pdfCitations: citations },
        }])
        loadChatSessions().catch(() => {})
        setIsOnline(true)
        return
      }

      const needsBrowserLocation = locationRequestPattern.test(cleanMessage)
      let chatLocation = browserLocation.location
      if (needsBrowserLocation && !chatLocation) {
        chatLocation = await requestSignedInLocation(user, { force: true })
        if (!chatLocation) {
          const answerCreatedAt = new Date().toISOString()
          setMessages((current) => [...current, {
            id: Date.now() + 1,
            role: 'assistant',
            text: LOCATION_PERMISSION_REQUIRED_MESSAGE,
            time: formatMessageTime(answerCreatedAt),
            createdAt: answerCreatedAt,
          }])
          setIsOnline(true)
          return
        }
      }
      const response = await apiFetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: cleanMessage, session_id: activeSessionId, ...(chatLocation ? { location: chatLocation } : {}), ...(selectedReply ? { reply_to_id: selectedReply.id } : {}) }),
        credentials: 'include',
      })
      if (!response.ok || !response.body) {
        const data = await response.json()
        throw new Error(data.detail || 'Nexa could not start the response stream.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let completedAnswer = ''
      let confirmationEmail = null
      let confirmationAction = null
      let skipChatMessage = false
      let researchRunId = ''
      let assistantMessageId = ''
      let streamFinished = false
      let lastLiveAnswerUpdateAt = 0

      while (!streamFinished) {
        const { value: chunk, done } = await reader.read()
        buffer += decoder.decode(chunk || new Uint8Array(), { stream: !done })
        const packets = buffer.split('\n\n')
        buffer = packets.pop() || ''

        for (const packet of packets) {
          const dataLine = packet.split('\n').find((line) => line.startsWith('data:'))
          if (!dataLine) continue
          const event = JSON.parse(dataLine.slice(5).trim())
          if (event.type === 'status') {
            setThinkingStatus(event.message)
            setThinkingDetail(event.detail || '')
            setThinkingEvents(() => {
              const nextEvent = {
                id: `${event.stage || 'step'}-${Date.now()}`,
                stage: event.stage || 'Update',
                message: event.message,
                detail: event.detail || '',
              }
              // The chat should communicate the live action, not expose a
              // running internal trace. Each status replaces the last one.
              return [nextEvent]
            })
          } else if (event.type === 'research_plan') {
            const labels = (event.sources || []).map((source) => source.label).filter(Boolean)
            const detail = labels.length
              ? `Selected sources: ${labels.join(', ')}.`
              : 'No research sources were selected.'
            setThinkingStatus('Research plan ready')
            setThinkingDetail(detail)
            setThinkingEvents([{ id: `research-plan-${Date.now()}`, stage: 'Research plan', message: 'Research plan ready', detail }])
          } else if (event.type === 'confirm_email') {
            confirmationEmail = event.email || null
            applyPendingEmail(event.email || null)
          } else if (event.type === 'confirm_mcp_action') {
            confirmationAction = event.action || null
            setPendingMcpAction(event.action || null)
          } else if (event.type === 'delta') {
            completedAnswer += event.content
            const now = Date.now()
            if (now - lastLiveAnswerUpdateAt >= 50) {
              lastLiveAnswerUpdateAt = now
              setLiveAnswer(completedAnswer)
            }
          } else if (event.type === 'error') {
            throw new Error(event.message)
          } else if (event.type === 'done') {
            completedAnswer = event.answer || ''
            skipChatMessage = Boolean(event.skip_chat)
            researchRunId = event.research_run_id || ''
            assistantMessageId = event.assistant_message_id || ''
            streamFinished = true
            break
          }
        }
        if (done) break
      }

      if (!completedAnswer.trim() && !confirmationEmail && !confirmationAction && !skipChatMessage) {
        throw new Error('Nexa returned an empty response.')
      }
      if (confirmationEmail || confirmationAction || skipChatMessage) {
        await loadPendingEmail()
        await loadPendingMcpAction()
      }
      if (!skipChatMessage && completedAnswer.trim()) {
        setLiveAnswer(completedAnswer)
        const answerCreatedAt = new Date().toISOString()
        setMessages((current) => [...current, {
          id: assistantMessageId || (Date.now() + 1),
          role: 'assistant',
          text: completedAnswer,
          time: formatMessageTime(answerCreatedAt),
          createdAt: answerCreatedAt,
          researchRunId,
          feedback: '',
        }])
        loadChatSessions().catch(() => {})
      }
      setIsOnline(true)
    } catch (error) {
      setApiError(error.message === 'Failed to fetch'
        ? 'Nexa API is offline. Start the backend and try again.'
        : error.message)
      setIsOnline(false)
    } finally {
      setIsThinking(false)
      setThinkingStatus('')
      setThinkingDetail('')
      setThinkingEvents([])
      setLiveAnswer('')
    }
  }, [
    input,
    isThinking,
    user,
    activeSessionId,
    chatSessions,
    pdfFile,
    loadPendingEmail,
    loadPendingMcpAction,
    applyPendingEmail,
    clearPdfAttachment,
    loadChatSessions,
    browserLocation,
    requestSignedInLocation,
    replyTarget,
    announceTyping,
  ])

  useEffect(() => {
    if (!user || !activeSessionId) return undefined
    const timer = window.setTimeout(() => {
      loadParticipants().catch(() => {
        setParticipants([])
        setSessionRole('member')
      })
    }, 0)
    return () => window.clearTimeout(timer)
  }, [activeSessionId, loadParticipants, user])

  useEffect(() => {
    if (!user || !activeSessionId) return undefined
    if (!activeSessionIsShared) return undefined
    let disposed = false
    let reconnectTimer = 0
    let reconnectDelay = 1000

    const recoverSession = async () => {
      const sessions = await loadChatSessions()
      if (sessions.length) await openChatSession(sessions[0].id)
      else await createChatSession()
    }

    const connect = () => {
      const apiUrl = new URL(API_BASE || window.location.origin, window.location.origin)
      apiUrl.protocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:'
      apiUrl.pathname = `/api/chats/${activeSessionId}/live`
      apiUrl.search = ''
      const authToken = readAuthToken()
      if (authToken) apiUrl.searchParams.set('auth_token', authToken)
      const socket = new WebSocket(apiUrl.toString())
      sessionSocketRef.current = socket

      socket.onopen = () => {
        reconnectDelay = 1000
      }

      socket.onmessage = async (packet) => {
        let event
        try { event = JSON.parse(packet.data) } catch { return }
        if (event.type === 'message' && event.message) {
          const incoming = messageWithDocumentReference(event.message, event.message.id)
          setMessages((current) => {
            if (current.some((message) => String(message.id) === String(incoming.id))) return current
            const optimisticIndex = [...current].reverse().findIndex((message) => (
              message.role === 'user' && message.text === incoming.text && !message.senderName && !message.sender_user_id
            ))
            if (optimisticIndex < 0) return [...current, incoming]
            const index = current.length - optimisticIndex - 1
            return current.map((message, itemIndex) => itemIndex === index ? incoming : message)
          })
          if (document.visibilityState === 'visible') markSessionRead(activeSessionId).catch(() => {})
          loadChatSessions().catch(() => {})
        } else if (event.type === 'typing' && event.user_id) {
          setTypingUsers((current) => {
            const existing = current.filter((person) => person.id !== event.user_id)
            return event.is_typing
              ? [...existing, { id: event.user_id, name: event.name || 'Someone' }]
              : existing
          })
        } else if (event.type === 'refresh') {
          try {
            await openChatSession(activeSessionId)
            await loadChatSessions()
            await loadParticipants(activeSessionId)
          } catch (error) {
            if (error.status === 404) await recoverSession()
          }
        } else if (event.type === 'error') {
          setApiError(event.message || 'Could not send the message.')
        }
      }
      socket.onclose = () => {
        if (!disposed) {
          reconnectTimer = window.setTimeout(connect, reconnectDelay)
          reconnectDelay = Math.min(reconnectDelay * 2, 30000)
        }
      }
    }

    connect()
    return () => {
      disposed = true
      window.clearTimeout(reconnectTimer)
      window.clearTimeout(typingStopTimerRef.current)
      typingActiveRef.current = false
      setTypingUsers([])
      sessionSocketRef.current?.close()
      sessionSocketRef.current = null
    }
  }, [activeSessionId, activeSessionIsShared, createChatSession, loadChatSessions, loadParticipants, markSessionRead, openChatSession, user])

  useEffect(() => {
    if (!user || !activeSessionId) return undefined
    const isSharedSession = Boolean(chatSessions.find((session) => session.id === activeSessionId)?.shared)
    if (!isSharedSession) return undefined
    let refreshing = false
    const refreshMessages = async () => {
      if (refreshing) return
      refreshing = true
      try {
        await openChatSession(activeSessionId, { markRead: false })
      } catch (error) {
        if (error.status === 404) setApiError('This chat session is no longer available.')
      } finally {
        refreshing = false
      }
    }
    const timer = window.setInterval(refreshMessages, 4000)
    return () => window.clearInterval(timer)
  }, [activeSessionId, chatSessions, openChatSession, user])

  useEffect(() => {
    if (!user) return undefined
    const refreshSessions = () => loadChatSessions().catch(() => {})
    const handleVisibility = () => {
      if (document.visibilityState === 'visible') {
        refreshSessions()
        if (activeSessionId) markSessionRead(activeSessionId).catch(() => {})
      }
    }
    const timer = window.setInterval(refreshSessions, 10000)
    window.addEventListener('focus', refreshSessions)
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener('focus', refreshSessions)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [activeSessionId, loadChatSessions, markSessionRead, user])

  useEffect(() => {
    if (!membersOpen && !composerHelpOpen) return undefined
    const closeOnOutsideClick = (event) => {
      if (membersPanelRef.current && !membersPanelRef.current.contains(event.target)) setMembersOpen(false)
      if (composerHelpRef.current && !composerHelpRef.current.contains(event.target)) setComposerHelpOpen(false)
    }
    window.addEventListener('mousedown', closeOnOutsideClick)
    return () => window.removeEventListener('mousedown', closeOnOutsideClick)
  }, [composerHelpOpen, membersOpen])

  useEffect(() => {
    const loadNexa = async () => {
      try {
        const params = new URLSearchParams(window.location.search)
        const authToken = params.get('token') || ''
        const googleConnectedService = params.get('google') === 'connected' ? params.get('service') || readPendingGoogleService() : ''
        const pendingGoogleService = readPendingGoogleService()
        if (pendingGoogleService) setGoogleActionBusy(pendingGoogleService)
        if (googleConnectedService) setGoogleActionBusy(googleConnectedService)
        if (authToken) writeAuthToken(authToken)
        const authResponse = await apiFetch(`${API_BASE}/api/auth/me`, { credentials: 'include' })
        if (authResponse.ok) {
          const auth = await authResponse.json()
          setUser(auth.user)
          if (await acceptInviteFromUrl()) return
          const sessions = await loadChatSessions()
          if (sessions.length) await openChatSession(sessions[0].id)
          else await createChatSession()
        } else if (inviteTokenFromPath()) {
          // Keep the invite URL intact; after sign-in the join effect redeems it.
          writePendingInviteToken(inviteTokenFromPath())
          setAuthView(true)
        }
        const [
          healthResponse,
          pendingEmailResponse,
          mcpServersResponse,
          pendingMcpResponse,
          googleServicesResponse,
          capabilitiesResponse,
        ] = await Promise.all([
          apiFetch(`${API_BASE}/api/health`, { credentials: 'include' }),
          apiFetch(`${API_BASE}/api/email/pending`, { credentials: 'include' }),
          apiFetch(`${API_BASE}/api/mcp/servers`, { credentials: 'include' }),
          apiFetch(`${API_BASE}/api/mcp/pending`, { credentials: 'include' }),
          apiFetch(`${API_BASE}/api/google/status`, { credentials: 'include' }),
          apiFetch(`${API_BASE}/api/capabilities`, { credentials: 'include' }),
        ])
        if (!healthResponse.ok) throw new Error()
        if (pendingEmailResponse.ok) {
          applyPendingEmail((await pendingEmailResponse.json()).pending_email || null)
        }
        if (mcpServersResponse.ok) {
          setMcpServers((await mcpServersResponse.json()).servers || [])
        }
        if (pendingMcpResponse.ok) {
          setPendingMcpAction((await pendingMcpResponse.json()).pending_action || null)
        }
        if (googleServicesResponse.ok) {
          const services = (await googleServicesResponse.json()).services || []
          setGoogleServices(mergeGoogleServices(services))
          const verifiedService = services.find((service) => service.service === googleConnectedService)
          if (googleConnectedService && !verifiedService?.connected) {
            setApiError(`Google completed the authorization redirect, but Nexa could not verify the ${googleConnectedService.replaceAll('_', ' ')} connection. Reconnect it from Settings.`)
          }
          writePendingGoogleService('')
        }
        if (capabilitiesResponse.ok) {
          setCapabilities(await capabilitiesResponse.json())
        }
        if (params.get('google') === 'error') {
          writePendingGoogleService('')
          setGoogleActionBusy('')
          setApiError(params.get('detail') || 'Google account connection was not completed.')
        }
        if (params.has('google') || params.has('auth') || params.has('token')) window.history.replaceState({}, '', window.location.pathname)
        setIsOnline(true)
        setGoogleActionBusy('')
      } catch {
        setIsOnline(false)
        setGoogleActionBusy('')
        setApiError('Nexa API is offline. Start the backend to begin chatting.')
      }
    }
    loadNexa()
  }, [acceptInviteFromUrl, applyPendingEmail, createChatSession, loadChatSessions, openChatSession])

  useEffect(() => {
    if (!user) {
      locationPromptedRef.current = ''
      const resetTimer = window.setTimeout(() => setBrowserLocation({
        status: 'idle',
        location: null,
        error: '',
        updatedAt: 0,
      }), 0)
      return () => window.clearTimeout(resetTimer)
    }
    const key = userLocationCacheKey(user)
    if (!key || locationPromptedRef.current === key) return
    locationPromptedRef.current = key
    requestSignedInLocation(user).catch(() => {
      // The helper stores a user-facing error state; no need to interrupt the app.
    })
  }, [user, requestSignedInLocation])

  useEffect(() => {
    const feed = feedRef.current
    if (!feed || !shouldStickToBottomRef.current) return undefined
    if (scrollFrameRef.current) window.cancelAnimationFrame(scrollFrameRef.current)
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      scrollFrameRef.current = 0
      feed.scrollTop = feed.scrollHeight
    })
    return () => {
      if (scrollFrameRef.current) {
        window.cancelAnimationFrame(scrollFrameRef.current)
        scrollFrameRef.current = 0
      }
    }
  }, [messages.length, isThinking, liveAnswer, thinkingStatus])

  if (authView) return <AuthScreen onSignedIn={completeSignIn} />

  const activeSession = chatSessions.find((session) => session.id === activeSessionId)
  const sharedSessionActive = activeSessionIsShared
  const showingJoinedInviteConfirmation = Boolean(joinedInviteSessionId && activeSessionId === joinedInviteSessionId)
  const agentMode = sharedSessionActive && hasAgentMention(input)
  const composerValue = agentMode ? stripAgentMention(input) : input
  const agentSuggestionVisible = sharedSessionActive && !agentMode && AGENT_PARTIAL_PATTERN.test(input.trimStart())
  const canSendMessage = agentMode ? Boolean(composerValue.trim()) : Boolean(input.trim())

  return (
    <main className={`app-shell theme-${theme}`}>
      <AmbientParticles />
      <header className="topbar">
        <a className="brand" href="/" aria-label="Nexa home">
          <span className="brand-emblem"><Logo /></span>
          <span className="brand-copy">
            <strong>Nexa</strong>

          </span>
        </a>

        <div className="system-state">
          <Sparkles size={14} />
          <span>{isOnline ? 'Workspace ready' : 'Reconnecting'}</span>
        </div>

        <div className="topbar-actions">
          <div className="navbar-google-actions" aria-label="Google connections">
            {['gmail', 'google_drive', 'google_calendar'].map((service) => {
              const details = googleServices.find((item) => item.service === service)
              const icon = service === 'google_calendar' ? 'calendar.png' : service === 'google_drive' ? 'drive.png' : 'gmail.png'
              const label = details?.label || (service === 'google_drive' ? 'Google Drive' : service === 'google_calendar' ? 'Google Calendar' : 'Gmail')
              return <button className={`navbar-google-button ${details?.connected ? 'connected' : ''}`} type="button" key={service} onClick={() => details?.connected ? disconnectGoogleService(service) : connectGoogleService(service)} disabled={googleActionBusy === service} title={details?.connected ? `${label} connected${details.email ? ` · ${details.email}` : ''}. Click to disconnect.` : `Connect ${label}`}><img src={`/${icon}`} alt="" /><span><b>{googleActionBusy === service ? '...' : label.replace('Google ', '')}</b><small>{details?.connected ? 'Connected' : 'Connect'}</small></span></button>
            })}
          </div>
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          {user ? <button className={`profile-button ${user.picture ? 'has-photo' : ''}`} type="button" data-email={user.email} title={user.email} aria-label={`Signed in as ${user.email}`}>{user.picture ? <img src={user.picture} alt="" referrerPolicy="no-referrer" /> : (user.name?.trim()?.charAt(0)?.toUpperCase() || 'U')}</button> : <button className="profile-button" type="button" onClick={() => setAuthView(true)} aria-label="Sign in">?</button>}
        </div>
      </header>

      <div className={`mobile-drawer-actions ${leftPanelOpen ? 'drawer-open' : ''}`} aria-label="Mobile panels">
        <button
          type="button"
          className={leftPanelOpen ? 'active' : ''}
          onClick={() => setLeftPanelOpen((open) => !open)}
          aria-label={leftPanelOpen ? 'Close chat sessions' : 'Open chat sessions'}
          aria-expanded={leftPanelOpen}
        >
          {leftPanelOpen ? <X size={20} /> : <Menu size={20} />}
          <span>Chats</span>
        </button>
      </div>

      <section className="command-layout">
        {leftPanelOpen && <button className="mobile-panel-scrim" type="button" aria-label="Close side panels" onClick={() => setLeftPanelOpen(false)} />}

        <aside className={`command-sidebar chat-sidebar ${leftPanelOpen ? 'mobile-open' : ''}`}>
          <AmbientParticles id="nexa-particles-sidebar-left" className="sidebar-particles" compact />
          <button className="mobile-drawer-close" type="button" onClick={() => setLeftPanelOpen(false)} aria-label="Close chat sessions"><X size={18} /></button>
          <div className="chat-sidebar-content">
            <button className="new-chat-button" type="button" disabled={activeSessionIsEmpty} onClick={() => openOrCreateDraftSession().catch((error) => setApiError(error.message))}><Plus size={17} /> New chat <span>⌘ K</span></button>
            <div className="chat-list-heading"><p className="section-label">CONVERSATIONS</p><button type="button" aria-label="Search chats"><Search size={15} /></button></div>
            <div className="chat-session-list">
              {chatSessions.map((session) => (
                <div className={`chat-session-row ${session.id === activeSessionId ? 'active' : ''} ${session.unread_count > 0 || session.has_unread ? 'has-unread' : ''}`} key={session.id}>
                  <button type="button" className={`chat-session-button ${session.id === activeSessionId ? 'active' : ''}`} onClick={() => { setJoinedInviteSessionId(''); openChatSession(session.id).catch((error) => setApiError(error.message)); setLeftPanelOpen(false) }}>
                    <span className="chat-session-title">{session.title}</span>
                    <span className="chat-session-meta">
                      {session.shared && <small className="session-member-count"><Users size={12} />{session.member_count || 2}</small>}
                      {(session.unread_count > 0 || session.has_unread) && <small className="session-unread-badge" aria-label={`${session.unread_count || 'Unread'} unread message${session.unread_count === 1 ? '' : 's'}`}>{session.unread_count > 9 ? '9+' : session.unread_count || '!'}</small>}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="chat-session-delete"
                    onClick={() => deleteChatSession(session.id)}
                    disabled={deletingSessionId === session.id}
                    aria-label={`Delete ${session.title}`}
                    title="Delete chat"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
              {!user && <p>Send a message to sign in and start saving chats.</p>}
            </div>
            <section className="connected-tools-mini" aria-label="Connected tools">
              <div className="connected-tools-mini-head">
                <p className="section-label">Available Plugins</p>
       
              </div>
              {mergeGoogleServices(googleServices).map((service) => {
                const isBusy = googleActionBusy === service.service || service.connecting
                const statusText = isBusy ? (service.connected ? 'Disconnecting...' : 'Connecting...') : service.connected ? 'Connected' : 'Not connected'
                return (
                  <button
                    className={`connected-tool-row ${service.connected ? 'connected' : ''} ${isBusy ? 'loading' : ''}`}
                    type="button"
                    key={service.service}
                    onClick={() => service.connected ? disconnectGoogleService(service.service) : connectGoogleService(service.service)}
                    disabled={isBusy}
                    title={service.connected ? `Disconnect ${service.label}${service.email ? ` (${service.email})` : ''}` : `Connect ${service.label}`}
                  >
                    <img src={`/${googleIconForService(service.service)}`} alt="" />
                    <span>
                      <strong>{compactGoogleLabel(service.label)}</strong>
                      <small>{service.email && service.connected ? service.email : statusText}</small>
                    </span>
                    <i className={service.connected ? 'connected' : ''} aria-label={statusText} />
                  </button>
                )
              })}
              {mcpServers.filter((server) => server.active && !server.oauth_service).slice(0, 3).map((server) => (
                <div className="connected-tool-row passive" key={server.name}>
                  <span className="tool-letter">{server.label.charAt(0)}</span>
                  <span><strong>{server.label}</strong><small>{server.read_only ? 'Read-only access' : 'Connected'}</small></span>
                  <i className="connected" aria-label="Connected" />
                </div>
              ))}
            </section>
          </div>
          {/* <div className="core-card core-card-bottom">
            <div className={`core-visual ${isThinking ? 'processing' : ''}`}>
              <span className="core-ring ring-one" />
              <span className="core-ring ring-two" />
              <span className="core-ring ring-three" />
              <span className="core-center"><Logo /></span>
            </div>
            <div className="core-copy">
              <span>NEXA CORE</span>
              <strong>{isThinking ? 'Processing request' : 'Ready when you are'}</strong>
            </div>
          </div> */}

          <div className="sidebar-foot">
            <span className="shield-icon" aria-hidden="true" />
            <p><strong>Private by default</strong><small>Chat data stays in your workspace</small></p>
          </div>
        </aside>

        <section className={`chat-stage ${showingJoinedInviteConfirmation ? 'invite-confirmation' : ''} ${isConversationScrolled ? 'conversation-scrolled' : ''}`} aria-label="Nexa conversation">

          {user && activeSessionId && (
            <div className={`session-collaboration-bar ${isConversationScrolled ? 'is-translucent' : ''}`}>
              <div>
                <span className="section-label">{activeSession?.shared ? 'SHARED' : 'PRIVATE'}</span>
                <strong>{activeSession?.title || 'New chat'}</strong>
              </div>
              {sessionDayLabel && <span className="session-day-label">{sessionDayLabel}</span>}
              {typingLabel && <span className="session-typing-indicator" role="status"><i />{typingLabel}</span>}
              <div className="session-collaboration-actions">
                <button type="button" className="session-members-button" onClick={() => { setMembersOpen((open) => !open); loadParticipants().catch((error) => setApiError(error.message)) }} title="Manage members" aria-label="Manage members">
                  {participants.length > 1 ? <span className="participant-avatars">{participants.slice(0, 3).map((participant) => participant.picture ? <img key={participant.user_id} src={participant.picture} alt="" referrerPolicy="no-referrer" /> : <i key={participant.user_id}>{participant.name?.trim()?.charAt(0)?.toUpperCase() || 'M'}</i>)}</span> : <Users size={17} />}
                  <span>{participants.length || 1}</span>
                </button>
                {sessionRole === 'admin' && <button type="button" onClick={shareChat} disabled={memberBusy === 'invite'} title="Copy invite link" aria-label="Copy invite link"><Share2 size={17} /></button>}
              </div>
              {membersOpen && (
                <section className="members-panel" ref={membersPanelRef} aria-label="Session members">
                  <div className="members-panel-heading"><strong>Members</strong>{sessionRole === 'admin' && <button type="button" onClick={shareChat} disabled={memberBusy === 'invite'}>{memberBusy === 'invite' ? 'Creating...' : 'Copy invite link'}</button>}</div>
                  {participants.map((participant) => (
                    <div className="member-row" key={participant.user_id}>
                      <div><strong>{participant.name}{participant.user_id === user.id ? ' (you)' : ''}</strong><small>{participant.email}</small></div>
                      <span className={`member-role ${participant.role}`}>{participant.role === 'admin' ? <Crown size={12} /> : null}{participant.role}</span>
                      {sessionRole === 'admin' && participant.user_id !== user.id && (
                        <div className="member-actions">
                          <button type="button" disabled={memberBusy === participant.user_id} onClick={() => changeParticipantRole(participant, participant.role === 'admin' ? 'member' : 'admin')} title={participant.role === 'admin' ? 'Make member' : 'Make admin'}><Crown size={14} /></button>
                          <button type="button" disabled={memberBusy === participant.user_id} onClick={() => removeParticipant(participant)} title="Remove member"><UserMinus size={14} /></button>
                        </div>
                      )}
                    </div>
                  ))}
                </section>
              )}
            </div>
          )}

          {shareDialogOpen && (
            <div className="share-dialog-overlay" role="presentation" onMouseDown={closeShareDialog}>
              <section className="share-dialog" role="dialog" aria-modal="true" aria-label="Share chat invite" onMouseDown={(event) => event.stopPropagation()}>
                <div className="share-dialog-head"><div><span className="section-label">SHARE SESSION</span><h2>Invite collaborators</h2></div><button type="button" onClick={closeShareDialog} aria-label="Close share dialog"><X size={18} /></button></div>
                <div className="share-history-options" role="group" aria-label="Conversation history sharing">
                  {shareHistoryOptions.map((option) => (
                    <label className="share-history-option" key={option.value}>
                      <input
                        type="radio"
                        name="share-history-mode"
                        checked={(shareHistoryMode || 'none') === option.value}
                        onChange={() => chooseShareHistoryMode(option.value)}
                      />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
                <div className="share-link-row"><Link2 size={20} /><input value={inviteLink} readOnly aria-label="Invite link" />{inviteLink ? <button type="button" onClick={copyInviteLink}><Copy size={15} />{inviteCopied ? 'Copied' : 'Copy'}</button> : <button type="button" disabled><Copy size={15} />Creating...</button>}</div>
                {inviteLink && <div className="share-app-links">
                  <a href={`https://wa.me/?text=${encodeURIComponent(`Join my Nexa chat: ${inviteLink}`)}`} target="_blank" rel="noreferrer"><img src="/whatsapp.png" alt="" />WhatsApp</a>
                  <a href={`https://t.me/share/url?url=${encodeURIComponent(inviteLink)}&text=${encodeURIComponent('Join my Nexa chat')}`} target="_blank" rel="noreferrer"><img src="/telegram.png" alt="" />Telegram</a>
                  <a href={`https://x.com/intent/post?text=${encodeURIComponent(`Join my Nexa chat: ${inviteLink}`)}`} target="_blank" rel="noreferrer"><img src="/X.png" alt="" />X</a>
                </div>}
              </section>
            </div>
          )}

          {pendingRejoinInvite && (
            <div className="share-dialog-overlay rejoin-dialog-overlay" role="presentation">
              <section className="share-dialog rejoin-dialog" role="dialog" aria-modal="true" aria-label="Rejoin grouped session">
                <div className="share-dialog-head">
                  <div>
                    <span className="section-label">REJOIN SESSION</span>
                    <h2>You are going to join the previous grouped session</h2>
                  </div>
                </div>
                <p className="rejoin-dialog-copy">{pendingRejoinInvite.sessionTitle}</p>
                <div className="share-history-options rejoin-options" role="radiogroup" aria-label="Private conversation sharing">
                  <button type="button" className="share-history-option rejoin-option" onClick={() => finishRejoinInvite(true)} disabled={Boolean(rejoinBusy)}>
                    <input type="radio" checked={rejoinBusy === 'share'} readOnly />
                    <span>{rejoinBusy === 'share' ? 'Sharing private conversation...' : 'Do you want to share your private conversation'}</span>
                  </button>
                  <button type="button" className="share-history-option rejoin-option" onClick={() => finishRejoinInvite(false)} disabled={Boolean(rejoinBusy)}>
                    <input type="radio" checked={rejoinBusy === 'private'} readOnly />
                    <span>{rejoinBusy === 'private' ? 'Keeping private conversation private...' : 'Do not share my conversation'}</span>
                  </button>
                </div>
              </section>
            </div>
          )}

          <div className="mobile-prompts" aria-label="Suggested prompts">
            {suggestions.map((suggestion) => (
              <button key={suggestion.label} type="button" onClick={() => sendMessage(suggestion.label)}>
                {suggestion.label}
              </button>
            ))}
          </div>

          <div className="conversation" ref={feedRef} aria-live="polite">
            {showingJoinedInviteConfirmation ? (
              <article className="message system invite-joined-message">
                <p className="system-message"><span className="system-message-pulse" aria-hidden="true" />You were added</p>
              </article>
            ) : activeSessionIsEmpty && <section className="empty-chat-hero">
              <div className="hero-orbit"><span /><Sparkles size={28} /></div>
              {/* <p className="section-label">NEXA WORKSPACE</p> */}
              <h2>Nexa - Shared AI.</h2>
              <p>Ask Nexa to research, compare, summarize, draft, or plan. Pick a starter below or type your own. Add people you like to share knowledge.</p>
              <div className="hero-pills" aria-label="Sample prompts">
                {suggestions.map(({ label, icon: Icon }) => (
                  <button key={label} type="button" onClick={() => sendMessage(label)} disabled={isThinking}>
                    <Icon size={14} />
                    {label}
                  </button>
                ))}
              </div>
            </section>}
            {!showingJoinedInviteConfirmation && !activeSessionIsEmpty && messages.map((message) => (
              <article
                className={`message ${message.role}`}
                key={message.id}
                {...(message.role !== 'system' && message.createdAt ? { 'data-message-day': formatMessageDay(message.createdAt) } : {})}
              >
                {message.role === 'system' ? <p className="system-message"><span className="system-message-pulse" aria-hidden="true" />{systemMessageText(message)}</p> : <>
                {isAssistantLikeMessage(message) && (
                  <div className="assistant-avatar"><Logo /></div>
                )}
                <div className="message-content">
                  <div className="message-meta">
                    <button
                      className="message-reply-button"
                      type="button"
                      onClick={() => startReply(message)}
                      aria-label={`Reply to ${messageAuthorLabel(message)}`}
                      title="Reply"
                    >
                      <Reply size={13} />
                    </button>
                    <strong>{messageAuthorLabel(message)}</strong>
                    <span>{message.time}</span>
                  </div>
                  {message.replyTo && (
                    <div className="message-reply-quote">
                      <strong>{message.replyTo.role === 'assistant' || message.replyTo.researchRunId ? 'Nexa' : (message.replyTo.senderName || (message.replyTo.sender_user_id === user?.id ? 'You' : 'Member'))}</strong>
                      <span>{message.replyTo.text}</span>
                    </div>
                  )}
                  {isAssistantLikeMessage(message)
                    ? (
                      <>
                        <MarkdownResponse>{message.text}</MarkdownResponse>
                        <AnswerCards message={message} sessionId={activeSessionId} />
                        <AssistantResponseActions
                          message={message}
                          canSaveFeedback={Boolean(activeSessionId && typeof message.id === 'string' && message.id.length > 20)}
                          onFeedback={submitAssistantFeedback}
                        />
                      </>
                    )
                    : (
                      <>
                        <p>{message.text}</p>
                        {message.documentName && (
                          <span className="message-document-ref" title={message.documentName}>
                            <FileText size={13} aria-hidden="true" />
                            {message.documentName}
                          </span>
                        )}
                      </>
                  )}
                </div>
                </>}
              </article>
            ))}

            {isThinking && (
              <article className="message assistant active-response">
                <div className="assistant-avatar"><Logo /></div>
                <div className="message-content stream-response">
                  <div className="thinking-stage" aria-live="polite">
                    <span className="thinking-pulse"><i /><i /><i /></span>
                    <span>{thinkingStatus || 'Thinking'}</span>
                  </div>
                  {thinkingDetail && <p className="thinking-detail">{thinkingDetail}</p>}
                  <div className="agent-steps-in-chat" aria-label="Current agent action">
                    <div className="agent-step">
                      <span className="agent-step-orb" aria-hidden="true" />
                      <strong>{thinkingStatus || 'Working on your request'}</strong>
                    </div>
                  </div>
                  {liveAnswer && <MarkdownResponse streaming>{liveAnswer}</MarkdownResponse>}
                </div>
              </article>
            )}
          </div>

          <footer className="composer-dock">
            {apiError && <p className="api-error" role="alert">{apiError}</p>}

            {pendingEmail && (
              <div className="email-confirmation-overlay" role="presentation">
                <section className="email-confirmation-card" aria-label="Email confirmation" role="dialog" aria-modal="true">
                  <div className="email-confirmation-head">
                    <div>
                      <p className="section-label">EMAIL CONFIRMATION</p>
                      <h3>Should I send this email?</h3>
                    </div>
                    <div className="email-confirmation-tools">
                      <span className="email-confirmation-pill">Awaiting approval</span>
                      <button
                        type="button"
                        className="email-close-button"
                        onClick={() => handlePendingEmailAction('cancel')}
                        disabled={Boolean(emailActionBusy)}
                        aria-label="Close email confirmation"
                        title="Close"
                      >
                        &times;
                      </button>
                    </div>
                  </div>

                  <div className="email-confirmation-grid">
                    <div className="email-confirmation-field">
                      <span>From</span>
                      <strong>{pendingEmail.sender}</strong>
                    </div>
                    <div className="email-confirmation-field">
                      <span>To</span>
                      <label className="email-inline-input">
                        <input
                          value={pendingRecipient}
                          onChange={(event) => setPendingRecipient(event.target.value)}
                          placeholder="name@company.com"
                          aria-label="Recipient email address"
                        />
                      </label>
                    </div>
                    <div className="email-confirmation-field">
                      <span>Subject</span>
                      <strong>{pendingEmail.subject}</strong>
                    </div>
                    <div className="email-confirmation-field">
                      <span>CC</span>
                      <label className="email-inline-input">
                        <input
                          value={pendingCc}
                          onChange={(event) => setPendingCc(event.target.value)}
                          placeholder="Optional"
                          aria-label="CC email address"
                        />
                      </label>
                    </div>
                    <div className="email-confirmation-field">
                      <span>BCC</span>
                      <label className="email-inline-input">
                        <input
                          value={pendingBcc}
                          onChange={(event) => setPendingBcc(event.target.value)}
                          placeholder="Optional"
                          aria-label="BCC email address"
                        />
                      </label>
                    </div>
                  </div>

                  <div className="email-preview-card">
                    <span>Email body</span>
                    <pre>{pendingEmail.body}</pre>
                  </div>

                  <div className="email-confirmation-actions">
                    <button
                      type="button"
                      className="email-send-button"
                      onClick={() => handlePendingEmailAction('confirm')}
                      disabled={Boolean(emailActionBusy)}
                    >
                      {emailActionBusy === 'confirm' ? 'Sending...' : 'Send email'}
                    </button>
                    <button
                      type="button"
                      className="email-cancel-button"
                      onClick={() => handlePendingEmailAction('cancel')}
                      disabled={Boolean(emailActionBusy)}
                    >
                      {emailActionBusy === 'cancel' ? 'Cancelling...' : 'Cancel'}
                    </button>
                  </div>
                </section>
              </div>
            )}

            {pendingMcpAction && (
              <div className="email-confirmation-overlay" role="presentation">
                <section className="email-confirmation-card" aria-label="Connected app confirmation" role="dialog" aria-modal="true">
                  <div className="email-confirmation-head">
                    <div>
                      <p className="section-label">CONNECTED APP APPROVAL</p>
                      <h3>Should I run this action?</h3>
                    </div>
                    <div className="email-confirmation-tools">
                      <span className="email-confirmation-pill">Awaiting approval</span>
                      <button
                        type="button"
                        className="email-close-button"
                        onClick={() => handlePendingMcpAction('cancel')}
                        disabled={Boolean(mcpActionBusy)}
                        aria-label="Close connected app confirmation"
                        title="Close"
                      >
                        &times;
                      </button>
                    </div>
                  </div>

                  <div className="email-confirmation-grid">
                    <div className="email-confirmation-field">
                      <span>Service</span>
                      <strong>{pendingMcpAction.server_label || pendingMcpAction.server_name}</strong>
                    </div>
                    <div className="email-confirmation-field">
                      <span>Action</span>
                      <strong>{pendingMcpAction.display_name}</strong>
                    </div>
                  </div>

                  <div className="email-preview-card">
                    <span>Arguments</span>
                    <pre>{JSON.stringify(pendingMcpAction.arguments || {}, null, 2)}</pre>
                  </div>

                  <div className="email-confirmation-actions">
                    <button
                      type="button"
                      className="email-send-button"
                      onClick={() => handlePendingMcpAction('confirm')}
                      disabled={Boolean(mcpActionBusy)}
                    >
                      {mcpActionBusy === 'confirm' ? 'Running...' : 'Confirm action'}
                    </button>
                    <button
                      type="button"
                      className="email-cancel-button"
                      onClick={() => handlePendingMcpAction('cancel')}
                      disabled={Boolean(mcpActionBusy)}
                    >
                      {mcpActionBusy === 'cancel' ? 'Cancelling...' : 'Cancel'}
                    </button>
                  </div>
                </section>
              </div>
            )}

            {pdfFile && (
              <div className="pdf-attachment-strip" role="status">
                <span className="pdf-file-icon" aria-hidden="true" />
                <div>
                  <strong>{pdfFile.name}</strong>
                  <small>Use /remember in prompt to save it; normal uploads are used once and never stored</small>
                </div>
                <button
                  type="button"
                  onClick={clearPdfAttachment}
                  disabled={isThinking}
                  aria-label="Remove attached PDF"
                  title="Remove PDF"
                >
                  &times;
                </button>
              </div>
            )}

            {replyTarget && (
              <div className="reply-composer-preview" role="status">
                <Reply size={14} aria-hidden="true" />
                <div>
                  <strong>Replying to {replyTarget.role === 'assistant' || replyTarget.researchRunId ? 'Nexa' : (replyTarget.senderName || (replyTarget.sender_user_id === user?.id ? 'you' : 'Member'))}</strong>
                  <span>{replyTarget.text}</span>
                </div>
                <button type="button" onClick={() => setReplyTarget(null)} aria-label="Cancel reply" title="Cancel reply">
                  <X size={14} />
                </button>
              </div>
            )}

            <div className="composer-workspace-hint">
              <Sparkles size={12} aria-hidden="true" />
              <span>Nexa can search across your connected workspace</span>
              <Command size={13} aria-hidden="true" />
            </div>
            <form className={`text-composer ${agentMode ? 'agent-mode' : ''}`} onSubmit={(event) => { event.preventDefault(); sendMessage() }}>
              <input
                ref={pdfInputRef}
                className="pdf-input"
                type="file"
                accept="application/pdf,.pdf"
                onChange={handlePdfChange}
                aria-label="Attach PDF"
              />
              <button
                className={`composer-attach ${pdfFile ? 'active' : ''}`}
                type="button"
                onClick={() => pdfInputRef.current?.click()}
                disabled={isThinking}
                aria-label={pdfFile ? 'Replace attached PDF' : 'Attach PDF'}
                title={pdfFile ? 'Replace PDF' : 'Attach PDF'}
              >
                <span className="attach-icon" aria-hidden="true" />
              </button>
              <label className={`composer-input ${agentMode ? 'agent-input' : ''}`}>
                {agentMode && (
                  <button
                    className="agent-composer-badge"
                    type="button"
                    onClick={clearAgentMention}
                    title="Remove agent mention"
                    aria-label="Remove agent mention"
                  >
                    <Logo />
                    <span>Nexa</span>
                    <X size={12} aria-hidden="true" />
                  </button>
                )}
                <input
                  ref={composerInputRef}
                  value={composerValue}
                  onChange={(event) => {
                    setInput(agentMode ? `@Nexa ${event.target.value}` : event.target.value)
                    announceTyping(Boolean(event.target.value.trim()))
                  }}
                  onKeyDown={(event) => {
                    if (agentSuggestionVisible && (event.key === 'Enter' || event.key === 'Tab' || event.key === 'ArrowDown')) {
                      event.preventDefault()
                      activateAgentMention()
                    } else if (agentMode && !composerValue && event.key === 'Backspace') {
                      event.preventDefault()
                      clearAgentMention()
                    }
                  }}
                  placeholder={agentMode ? 'Ask Nexa to handle this...' : pdfFile ? 'Ask about the attached PDF' : sharedSessionActive ? 'Message members or start with @Nexa' : 'Message Nexa'}
                  aria-label="Message Nexa"
                />
                {agentSuggestionVisible && (
                  <button
                    className="agent-mention-suggestion"
                    type="button"
                    onClick={activateAgentMention}
                    aria-label="Use agent mention"
                  >
                    <span className="agent-mention-suggestion-logo"><Logo /></span>
                    <span className="agent-mention-suggestion-copy">
                      <strong>@Nexa</strong>
                      <small>Ask Nexa to respond in this shared chat</small>
                    </span>
                  </button>
                )}
                <small>{agentMode ? 'Nexa will respond to this request' : pdfFile ? 'Prefix with /remember to save this document permanently' : sharedSessionActive ? 'Messages go to session members. Use @Nexa to call Nexa.' : 'Nexa will respond in this private session.'}</small>
              </label>
              <div className="composer-help-shell" ref={composerHelpRef}>
                <button
                  className={`composer-help-button ${composerHelpOpen ? 'active' : ''}`}
                  type="button"
                  onClick={() => setComposerHelpOpen((open) => !open)}
                  aria-label="Open chat help"
                  aria-expanded={composerHelpOpen}
                  title="Chat help"
                >
                  /
                </button>
                {composerHelpOpen && (
                  <section className="composer-help-panel" aria-label="Chat help">
                    <div className="composer-help-heading">
                      <CircleHelp size={16} />
                      <strong>Chat commands</strong>
                    </div>
                    <div className="composer-help-list">
                      <button type="button" onClick={() => insertComposerCommand('@Nexa ')}>
                        <code>@Nexa</code>
                        <span>Call Nexa in chats with multiple members.</span>
                      </button>
                      {!sharedSessionActive && (
                        <div className="composer-help-row">
                          <code>Private</code>
                          <span>This session sends every message to Nexa automatically.</span>
                        </div>
                      )}
                      <button type="button" onClick={() => insertComposerCommand(sharedSessionActive ? '@Nexa /remember ' : '/remember ')}>
                        <code>/remember</code>
                        <span>Save an attached PDF into your document memory.</span>
                      </button>
                      <button type="button" onClick={() => insertComposerCommand(sharedSessionActive ? '@Nexa /doc ' : '/doc ')}>
                        <code>/doc</code>
                        <span>Search your saved documents and answer from them.</span>
                      </button>
                      <button type="button" onClick={() => insertComposerCommand(sharedSessionActive ? '@Nexa /research ' : '/research ')}>
                        <code>/research</code>
                        <span>Run a cited deep-research report using relevant public sources.</span>
                      </button>
                      <div className="composer-help-row">
                        <code>Reply</code>
                        <span>Reply to a message, then ask Nexa to summarize, draft, or fact-check it.</span>
                      </div>
                      <div className="composer-help-row">
                        <code>Private tools</code>
                        <span>Gmail, Drive, and Calendar results stay private in shared chats.</span>
                      </div>
                    </div>
                  </section>
                )}
              </div>
              <button
                className="send-button"
                type="submit"
                disabled={!canSendMessage || isThinking}
                aria-label="Send message"
                title="Send"
              >
                <SendHorizontal size={18} strokeWidth={2.35} aria-hidden="true" />
              </button>
            </form>
            <div className="creator-links" aria-label="Creator links">
              <a className="creator-link-text" href="https://www.linkedin.com/in/yashraj-gupta-ai-fullstack-engineer/" target="_blank" rel="noreferrer">Know about creator</a>
              <a href="https://www.linkedin.com/in/yashraj-gupta-ai-fullstack-engineer/" target="_blank" rel="noreferrer" aria-label="Creator LinkedIn"><img src="/linkedin.png" alt="" /></a>
              <a href="https://leetcode.com/u/yashraj-ai-fullstack-engineer/" target="_blank" rel="noreferrer" aria-label="Creator LeetCode"><img src="/leetcode.png" alt="" /></a>
              <a href="https://blog-ai-3m6a.vercel.app/" target="_blank" rel="noreferrer" aria-label="Creator blog"><img src="/blog.png" alt="" /></a>
              <a href="https://github.com/yashraj-ai-fullstack-engineer" target="_blank" rel="noreferrer" aria-label="Creator GitHub"><img src="/github.png" alt="" /></a>
              <a href="mailto:yashrajgupta306@gmail.com" aria-label="Email creator"><img src="/gmail.png" alt="" /></a>
            </div>

          </footer>
        </section>
        <aside className="activity-rail">
          <AmbientParticles id="nexa-particles-sidebar-right" className="sidebar-particles" compact />
          <div className="rail-heading">
            <div>
              <p className="section-label">AGENT ACTIVITY</p>
              <h2>Live trace</h2>
            </div>
            <span className={`trace-state ${isThinking ? 'active' : ''}`}>
              {isThinking ? 'LIVE' : 'IDLE'}
            </span>
          </div>

          <div className="activity-card">
            <div className="activity-current">
              <span className={`activity-orb ${isThinking ? 'active' : ''}`} />
              <div>
                <small>CURRENT OPERATION</small>
                <strong>{thinkingStatus || 'Standing by'}</strong>
              </div>
            </div>

            {thinkingEvents.length > 0 ? (
              <div className="activity-timeline">
                {thinkingEvents.map((event, index) => (
                  <div className="activity-event" key={event.id}>
                    <span className={`timeline-node ${index === thinkingEvents.length - 1 && isThinking ? 'current' : ''}`} />
                    <div>
                      <small>{event.stage}</small>
                      <strong>{event.message}</strong>
                      {event.detail && <p>{event.detail}</p>}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="activity-idle">
                <span className="idle-line" />
                <p>Agent decisions and tool calls will appear here in real time.</p>
              </div>
            )}
          </div>

          <div className="connected-apps-card">
            <div className="connected-apps-heading">
              <div>
                <p className="section-label">CONNECTED APPS</p>
                <h3>Google workspace</h3>
              </div>
              <span className="connected-apps-badge">
                {googleServices.filter((service) => service.connected).length}/{Math.max(googleServices.length, 3)}
              </span>
            </div>

            <div className="connected-apps-list">
              {googleServices.map((service) => (
                <div className="google-service-card" key={service.service}>
                  <div className="google-service-topline">
                    <div className="google-service-copy">
                      <strong>{service.label}</strong>
                      <small>{service.connected ? `Connected${service.email ? ` · ${service.email}` : ''}` : 'Not connected'}</small>
                    </div>
                    <span className={`google-service-state ${service.connected ? 'connected' : ''}`}>
                      {service.connected ? 'Active' : 'Ready'}
                    </span>
                  </div>

                  <div className="google-service-actions">
                    {service.connected ? (
                      <button
                        className="google-service-button disconnect"
                        type="button"
                        onClick={() => disconnectGoogleService(service.service)}
                        disabled={googleActionBusy === service.service}
                      >
                        {googleActionBusy === service.service ? 'Disconnecting...' : 'Disconnect'}
                      </button>
                    ) : (
                      <button
                        className="google-service-button"
                        type="button"
                        onClick={() => connectGoogleService(service.service)}
                        title={service.configured ? `Connect ${service.label}` : `Connect ${service.label} (setup will be validated by the backend)`}
                      >
                        Connect {service.label}
                      </button>
                    )}
                  </div>
                </div>
              ))}

              {googleServices.length === 0 && (
                <div className="google-service-card google-service-empty">
                  <div className="google-service-copy">
                    <strong>Google setup needed</strong>
                    <small>Add OAuth values to the backend `.env` to connect Gmail, Calendar, and Drive.</small>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* <div className="system-readout">
            <p className="section-label">SYSTEM</p>
            <div><span>Orchestrator</span><strong>LangGraph</strong></div>
            <div><span>Reasoning</span><strong>Qwen</strong></div>
            <div><span>Search</span><strong className="ready">Ready</strong></div>
          </div> */}

          <div className="capability-stack">
            <p className="section-label">CAPABILITIES</p>
            {capabilities.local.map((capability) => (
              <div className="capability" key={capability.id}>
                <span
                  className={`capability-icon ${
                    capability.id === 'web'
                      ? 'search-icon'
                      : capability.id === 'email'
                          ? 'mail-icon'
                          : 'action-icon'
                  }`}
                  aria-hidden="true"
                />
                <div>
                  <strong>{capability.label}</strong>
                  <small>{capability.description}</small>
                </div>
              </div>
            ))}
          </div>

          <div className="capability-stack">
            <p className="section-label">CONNECTED TOOLS</p>
            {googleServices.map((service) => (
              <div className="capability google-service" key={service.service}>
                <span className="capability-icon mail-icon" aria-hidden="true" />
                <div>
                  <strong>{service.label}</strong>
                  <small>{service.connected ? `Connected${service.email ? ` · ${service.email}` : ''}` : 'Not connected'}</small>
                </div>
                {service.connected ? (
                  <button
                    className="google-service-button disconnect"
                    type="button"
                    onClick={() => disconnectGoogleService(service.service)}
                    disabled={googleActionBusy === service.service}
                  >
                    {googleActionBusy === service.service ? '...' : 'Disconnect'}
                  </button>
                ) : (
                  <button
                    className="google-service-button"
                    type="button"
                    onClick={() => connectGoogleService(service.service)}
                    title={service.configured ? `Connect ${service.label}` : `Connect ${service.label} (setup will be validated by the backend)`}
                  >
                    Connect
                  </button>
                )}
              </div>
            ))}
            {googleServices.length === 0 && (
              <div className="capability">
                <span className="capability-icon action-icon" aria-hidden="true" />
                <div><strong>Google setup needed</strong><small>Add OAuth values to the backend .env to connect apps</small></div>
              </div>
            )}
            {mcpServers.filter((server) => server.active && !server.oauth_service).map((server) => (
              <div className="capability" key={server.name}>
                <span className="capability-icon action-icon" aria-hidden="true" />
                <div><strong>{server.label}</strong><small>{server.read_only ? 'Read-only access' : 'Interactive access'}</small></div>
              </div>
            ))}
          </div>

          <p className="rail-footer">
            <span className="lock-icon" aria-hidden="true" />
            Local agent - Private session
          </p>
        </aside>
      </section>
    </main>
  )
}

export default App
