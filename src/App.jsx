import React, { useState, useRef, useCallback } from 'react'

// ============================================================
// CONFIG — Change this to your deployed backend URL
// ============================================================
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ============================================================
// STYLES
// ============================================================
const styles = `
  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0a0f1c;
    --bg-card: #111827;
    --bg-card-hover: #1a2235;
    --border: #1e293b;
    --border-active: #3b82f6;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.15);
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --normal: #10b981;
    --mild: #f59e0b;
    --severe: #ef4444;
    --font: 'DM Sans', -apple-system, sans-serif;
    --mono: 'JetBrains Mono', monospace;
    --radius: 12px;
    --radius-sm: 8px;
  }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  .app {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 60px;
  }

  /* ---- HEADER ---- */
  .header {
    text-align: center;
    margin-bottom: 40px;
    padding-bottom: 32px;
    border-bottom: 1px solid var(--border);
  }
  .header-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-glow);
    border: 1px solid rgba(59,130,246,0.25);
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
  }
  .header h1 {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
  }
  .header h1 span { color: var(--accent); }
  .header p {
    color: var(--text-muted);
    font-size: 15px;
    max-width: 600px;
    margin: 0 auto;
  }

  /* ---- LAYOUT ---- */
  .main-grid {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 28px;
    align-items: start;
  }
  @media (max-width: 840px) {
    .main-grid { grid-template-columns: 1fr; }
  }

  /* ---- CARD ---- */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
  }
  .card-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 16px;
  }

  /* ---- UPLOAD ZONE ---- */
  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: transparent;
    position: relative;
    overflow: hidden;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    background: var(--accent-glow);
  }
  .upload-zone.has-image {
    padding: 0;
    border-style: solid;
    border-color: var(--border);
  }
  .upload-zone img {
    width: 100%;
    height: 260px;
    object-fit: contain;
    display: block;
    border-radius: 10px;
  }
  .upload-icon {
    width: 48px;
    height: 48px;
    margin: 0 auto 12px;
    opacity: 0.4;
  }
  .upload-text { color: var(--text-muted); font-size: 14px; }
  .upload-text strong { color: var(--text); }
  .upload-hint {
    font-size: 12px;
    color: var(--text-dim);
    margin-top: 8px;
  }

  /* ---- BUTTONS ---- */
  .btn {
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: var(--radius-sm);
    font-family: var(--font);
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .btn-primary {
    background: var(--accent);
    color: white;
  }
  .btn-primary:hover:not(:disabled) {
    background: #2563eb;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.3);
  }
  .btn-primary:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .btn-outline {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    font-size: 13px;
    padding: 10px;
  }
  .btn-outline:hover {
    border-color: var(--text-dim);
    color: var(--text);
  }

  /* ---- LOADING ---- */
  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid rgba(255,255,255,0.2);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---- RESULTS ---- */
  .results-panel { display: flex; flex-direction: column; gap: 20px; }

  .score-display {
    text-align: center;
    padding: 28px;
  }
  .score-number {
    font-family: var(--mono);
    font-size: 56px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
  }
  .score-label {
    font-size: 13px;
    color: var(--text-dim);
    letter-spacing: 0.5px;
  }
  .severity-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin-top: 12px;
  }
  .severity-Normal { background: rgba(16,185,129,0.15); color: var(--normal); border: 1px solid rgba(16,185,129,0.3); }
  .severity-Mild { background: rgba(245,158,11,0.15); color: var(--mild); border: 1px solid rgba(245,158,11,0.3); }
  .severity-Severe { background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }

  /* ---- SCORE BAR ---- */
  .score-bar-container { padding: 0 4px; margin-top: 20px; }
  .score-bar-track {
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    position: relative;
    overflow: hidden;
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
  }
  .score-bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 6px;
  }

  /* ---- CONFIDENCE BARS ---- */
  .confidence-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }
  .confidence-label {
    width: 60px;
    font-size: 13px;
    font-weight: 500;
    flex-shrink: 0;
  }
  .confidence-bar-track {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  .confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
  }
  .confidence-value {
    width: 48px;
    text-align: right;
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text-muted);
  }

  /* ---- IMAGE GRID ---- */
  .image-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .image-box {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }
  .image-box-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    padding: 10px 12px;
    background: rgba(0,0,0,0.2);
  }
  .image-box img {
    width: 100%;
    height: auto;
    min-height: 180px;
    object-fit: contain;
    display: block;
    background: #000;
    padding: 8px;
  }
    
  /* ---- EXPLANATION ---- */
  .explanation {
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-muted);
  }

  /* ---- ERROR ---- */
  .error-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    color: #fca5a5;
    font-size: 14px;
  }

  /* ---- EMPTY STATE ---- */
  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-dim);
  }
  .empty-state-icon { font-size: 40px; margin-bottom: 12px; opacity: 0.3; }
  .empty-state p { font-size: 14px; }

  /* ---- FOOTER ---- */
  .disclaimer {
    margin-top: 40px;
    padding: 16px 20px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-dim);
    text-align: center;
    line-height: 1.6;
  }
  .footer {
    text-align: center;
    margin-top: 24px;
    font-size: 12px;
    color: var(--text-dim);
  }
`

// ============================================================
// COMPONENTS
// ============================================================

function UploadIcon() {
  return (
    <svg className="upload-icon" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M8 32l8-8 6 6 8-10 10 12" strokeLinecap="round" strokeLinejoin="round"/>
      <rect x="4" y="6" width="40" height="36" rx="4" strokeLinecap="round"/>
      <circle cx="16" cy="16" r="3"/>
    </svg>
  )
}

function ScoreBar({ score, maxScore = 3 }) {
  const pct = Math.min((score / maxScore) * 100, 100)
  const color = score < 0.6 ? 'var(--normal)' : score < 1.75 ? 'var(--mild)' : 'var(--danger)'
  return (
    <div className="score-bar-container">
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="score-bar-labels">
        <span>0 (Normal)</span>
        <span>1.0</span>
        <span>2.0</span>
        <span>3 (Severe)</span>
      </div>
    </div>
  )
}

function ConfidenceBars({ probabilities }) {
  const items = [
    { label: 'Normal', value: probabilities.Normal, color: 'var(--normal)' },
    { label: 'Mild', value: probabilities.Mild, color: 'var(--mild)' },
    { label: 'Severe', value: probabilities.Severe, color: 'var(--danger)' },
  ]
  return (
    <div>
      {items.map(({ label, value, color }) => (
        <div key={label} className="confidence-row">
          <span className="confidence-label">{label}</span>
          <div className="confidence-bar-track">
            <div className="confidence-bar-fill" style={{ width: `${value}%`, background: color }} />
          </div>
          <span className="confidence-value">{value.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG only).')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File is too large. Maximum size is 10MB.')
      return
    }
    setError(null)
    setResult(null)
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const handlePredict = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to connect to the server. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const scoreColor = result
    ? result.score < 0.6 ? 'var(--normal)' : result.score < 1.75 ? 'var(--mild)' : 'var(--danger)'
    : 'var(--text)'

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="header-badge">AI-Powered Diagnostic Tool</div>
          <h1>Liver<span>Xplain</span></h1>
          <p>Explainable Vision Transformer for continuous fatty liver severity prediction from B-mode ultrasound images</p>
        </header>

        {/* Main Grid */}
        <div className="main-grid">
          {/* Left: Upload Panel */}
          <div>
            <div className="card">
              <div className="card-title">Upload Ultrasound Image</div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => handleFile(e.target.files[0])}
                style={{ display: 'none' }}
              />

              <div
                className={`upload-zone ${dragOver ? 'dragover' : ''} ${previewUrl ? 'has-image' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" />
                ) : (
                  <>
                    <UploadIcon />
                    <div className="upload-text">
                      <strong>Click to upload</strong> or drag and drop
                    </div>
                    <div className="upload-hint">JPG, PNG — max 10MB — liver ultrasound only</div>
                  </>
                )}
              </div>

              <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
                <button
                  className="btn btn-primary"
                  onClick={handlePredict}
                  disabled={!selectedFile || loading}
                >
                  {loading ? (
                    <><div className="spinner" /> Analyzing...</>
                  ) : (
                    '🔍 Predict Severity'
                  )}
                </button>
              </div>

              {selectedFile && (
                <button className="btn btn-outline" style={{ marginTop: 10 }} onClick={handleReset}>
                  Clear &amp; Upload New Image
                </button>
              )}

              {error && (
                <div className="error-box" style={{ marginTop: 16 }}>⚠️ {error}</div>
              )}
            </div>
          </div>

          {/* Right: Results Panel */}
          <div className="results-panel">
            {!result && !loading && (
              <div className="card empty-state">
                <div className="empty-state-icon">🔬</div>
                <p>Upload a liver ultrasound image and click "Predict" to see results</p>
              </div>
            )}

            {loading && (
              <div className="card empty-state">
                <div className="spinner" style={{ margin: '0 auto 16px', borderColor: 'var(--border)', borderTopColor: 'var(--accent)' }} />
                <p>Analyzing ultrasound image...</p>
                <p style={{ fontSize: 12, marginTop: 4 }}>Running ViT model + generating attention heatmap</p>
              </div>
            )}

            {result && (
              <>
                {/* Score Card */}
                <div className="card score-display">
                  <div className="score-number" style={{ color: scoreColor }}>
                    {result.score.toFixed(2)}
                  </div>
                  <div className="score-label">Severity Score (0–3 scale)</div>
                  <div className={`severity-badge severity-${result.prediction}`}>
                    {result.prediction}
                  </div>
                  <ScoreBar score={result.score} />
                </div>

                {/* Confidence Bars */}
                <div className="card">
                  <div className="card-title">Class Probabilities</div>
                  <ConfidenceBars probabilities={result.probabilities} />
                </div>

                {/* Images: Preprocessed + Heatmap */}
                <div className="card">
                  <div className="card-title">Attention Analysis</div>
                  <div className="image-grid">
                    <div className="image-box">
                      <div className="image-box-label">Preprocessed Input</div>
                      <img src={`data:image/png;base64,${result.preprocessed}`} alt="Preprocessed" />
                    </div>
                    <div className="image-box">
                      <div className="image-box-label">Attention Heatmap</div>
                      <img src={`data:image/png;base64,${result.heatmap}`} alt="Heatmap" />
                    </div>
                  </div>
                </div>

                {/* Explanation */}
                <div className="card">
                  <div className="card-title">Clinical Interpretation</div>
                  <div className="explanation">{result.explanation}</div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="disclaimer">
          ⚠️ <strong>Research Prototype Only</strong> — This system is developed for academic evaluation purposes.
          It is NOT approved for clinical diagnosis. Always consult a qualified radiologist for medical decisions.
        </div>

        <div className="footer">
          LiverXplain — Gavesh Dissanayake (w1870605) — IIT / University of Westminster — Supervised by Mr. Mahfoos Ahamed
        </div>
      </div>
    </>
  )
}
