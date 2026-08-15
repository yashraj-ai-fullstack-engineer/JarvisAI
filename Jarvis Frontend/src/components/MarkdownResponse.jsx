import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './MarkdownResponse.css'

function ExternalLink({ node, ...props }) {
  void node
  return <a {...props} target="_blank" rel="noopener noreferrer" />
}

function ResponsiveTable({ node, ...props }) {
  void node
  return <div className="markdown-response__table-wrap"><table {...props} /></div>
}

const components = {
  a: ExternalLink,
  table: ResponsiveTable,
}

/** Renders every agent response with one predictable Markdown presentation. */
const MarkdownResponse = memo(function MarkdownResponse({ children, streaming = false }) {
  return (
    <div className={`markdown-response${streaming ? ' markdown-response--streaming' : ''}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} skipHtml>
        {children}
      </ReactMarkdown>
      {streaming && <span className="markdown-response__cursor" aria-hidden="true" />}
    </div>
  )
})

export default MarkdownResponse
