import React, { forwardRef } from 'react';

// A4 clinical report. Everything printed here comes from the API response --
// the previous version hard-coded "ResNet-18" in the narrative while the backend
// was serving ResNet-50, and described the scan as a "radiograph" when it is an
// optical photograph.
const S = {
    h3: { fontSize: '14px', fontWeight: 'bold', borderBottom: '1px solid black', marginBottom: '12px', paddingBottom: '3px' } as React.CSSProperties,
    p: { fontSize: '12.5px', marginBottom: '12px', textAlign: 'justify' } as React.CSSProperties,
    td: { border: '1px solid #999', padding: '5px 8px', fontSize: '11.5px' } as React.CSSProperties,
    th: { border: '1px solid #999', padding: '5px 8px', fontSize: '11px', background: '#eee', textAlign: 'left', fontWeight: 'bold' } as React.CSSProperties,
};

const pct = (v: number) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

const PrintableReport = forwardRef(({ result, reportId }: any, ref: any) => {
    if (!result) return null;

    const now = new Date();
    const date = now.toLocaleDateString('en-GB', { year: 'numeric', month: 'long', day: 'numeric' });
    const time = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    const id = reportId || 'DFU-PREVIEW';
    const att = result.attention || {};
    const iwgdf = result.iwgdf;
    const model = result.model || {};
    const steps: string[] = result.report.triage_steps || result.report.triage.split('\n');

    return (
        <div ref={ref} className="p-10" style={{
            fontFamily: '"Times New Roman", Times, serif', color: '#000', backgroundColor: '#fff',
            WebkitPrintColorAdjust: 'exact', printColorAdjust: 'exact', lineHeight: 1.5, maxWidth: '850px', margin: '0 auto',
        }}>
            <div style={{ borderBottom: '2px solid black', paddingBottom: '10px', marginBottom: '22px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                <div>
                    <h1 style={{ fontSize: '25px', fontWeight: 'bold', margin: '0 0 4px 0' }}>ClinicalVision Diagnostic Laboratory</h1>
                    <p style={{ fontSize: '11.5px', fontStyle: 'italic', margin: 0 }}>Automated Neural Assessment &amp; Triage System — Research Prototype</p>
                </div>
                <div style={{ textAlign: 'right', fontSize: '11.5px' }}>
                    <p style={{ margin: '0 0 2px 0' }}><strong>Record ID:</strong> {id}</p>
                    <p style={{ margin: '0 0 2px 0' }}><strong>Date:</strong> {date}</p>
                    <p style={{ margin: 0 }}><strong>Time:</strong> {time}</p>
                </div>
            </div>

            <div style={{ textAlign: 'center', marginBottom: '26px' }}>
                <h2 style={{ fontSize: '17px', textDecoration: 'underline', marginBottom: '10px' }}>
                    DIAGNOSTIC REPORT: OPTICAL PLANTAR SCAN
                </h2>
                <div style={{ padding: '9px 18px', border: '2px solid black', display: 'inline-block', fontWeight: 'bold', fontSize: '15px', backgroundColor: '#f5f5f5' }}>
                    CLASSIFICATION: {result.risk} RISK — fused probability {pct(result.risk_probability)}
                </div>
            </div>

            <div style={{ marginBottom: '24px' }}>
                <h3 style={S.h3}>I. Quantitative Findings</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '10px' }}>
                    <tbody>
                        <tr>
                            <th style={S.th}>Image probability P(ulcer)</th><td style={S.td}>{pct(result.image_probability)}</td>
                            <th style={S.th}>Classifier confidence</th><td style={S.td}>{pct(result.confidence)}</td>
                        </tr>
                        <tr>
                            <th style={S.th}>Fused risk</th><td style={S.td}>{pct(result.risk_probability)} ({result.risk})</td>
                            <th style={S.th}>Clinical log-odds shift</th><td style={S.td}>{result.clinical_logit_shift >= 0 ? '+' : ''}{result.clinical_logit_shift?.toFixed(3)}</td>
                        </tr>
                        <tr>
                            <th style={S.th}>Attention area</th><td style={S.td}>{att.attention_area_pct}% of field</td>
                            <th style={S.th}>Discrete regions</th><td style={S.td}>{att.attention_regions}</td>
                        </tr>
                        <tr>
                            <th style={S.th}>Spatial focality</th><td style={S.td}>{att.focality?.toFixed(3)}</td>
                            <th style={S.th}>Peak activation</th><td style={S.td}>{att.peak_intensity?.toFixed(3)}</td>
                        </tr>
                        <tr>
                            <th style={S.th}>Model</th><td style={S.td}>{model.architecture} @ {model.input_size}px</td>
                            <th style={S.th}>Calibration T</th><td style={S.td}>{model.temperature?.toFixed(3)}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div style={{ marginBottom: '24px' }}>
                <h3 style={S.h3}>II. Clinical Narrative &amp; Assessment</h3>
                <p style={S.p}>{result.report.clinical_assessment}</p>
                <p style={S.p}>{result.report.visual_analysis}</p>
            </div>

            <div style={{ marginBottom: '24px' }}>
                <h3 style={S.h3}>III. Visual Evidence &amp; Feature Attribution</h3>
                <div style={{ display: 'flex', gap: '26px', alignItems: 'flex-start' }}>
                    <div style={{ flex: '0 0 38%' }}>
                        <div style={{ border: '2px solid black', padding: '3px' }}>
                            <img src={`data:image/png;base64,${result.overlay || result.heatmap}`}
                                alt="Grad-CAM attribution overlay"
                                style={{ width: '100%', height: 'auto', display: 'block' }} />
                        </div>
                        <p style={{ textAlign: 'center', fontSize: '10.5px', fontStyle: 'italic', marginTop: '6px' }}>
                            Fig 1: {result.cam_mode === 'gradcam++' ? 'Grad-CAM++' : 'Grad-CAM'} attribution for the ulcer class (warm = higher activation)
                        </p>
                    </div>
                    <div style={{ flex: 1 }}>
                        <p style={{ fontSize: '12px', marginBottom: '9px' }}>
                            Exact Shapley decomposition of the fused log-odds. Positive values raise
                            the estimated risk relative to the population reference profile:
                        </p>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr><th style={S.th}>Factor</th><th style={S.th}>Value</th><th style={S.th}>Log-odds</th><th style={S.th}>Odds ratio</th><th style={S.th}>Influence</th></tr>
                            </thead>
                            <tbody>
                                {(result.attribution || []).map((a: any, i: number) => (
                                    <tr key={i}>
                                        <td style={S.td}>{a.feature}</td>
                                        <td style={S.td}>{a.value}{a.unit ? ` ${a.unit}` : ''}</td>
                                        <td style={S.td}>{a.logit_contribution >= 0 ? '+' : ''}{a.logit_contribution?.toFixed(3)}</td>
                                        <td style={S.td}>{a.odds_ratio?.toFixed(2)}</td>
                                        <td style={S.td}>{a.influence_pct}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div style={{ marginBottom: '24px', pageBreakBefore: 'always' }}>
                <h3 style={S.h3}>IV. Risk Stratification &amp; Action Plan</h3>
                {iwgdf && (
                    <p style={{ ...S.p, border: '1px solid #999', padding: '9px', background: '#f7f7f7' }}>
                        <strong>IWGDF 2023 risk category {iwgdf.category} — {iwgdf.label}.</strong>{' '}
                        Basis: {iwgdf.basis}. Recommended screening interval: <strong>{iwgdf.screening_interval}</strong>.
                    </p>
                )}
                <p style={{ fontSize: '12.5px', marginBottom: '8px' }}>
                    Pursuant to the {result.risk} risk classification, the system recommends:
                </p>
                <ol style={{ fontSize: '12.5px', paddingLeft: '26px', margin: 0 }}>
                    {steps.map((s, i) => <li key={i} style={{ marginBottom: '6px' }}>{s.replace(/^\d+\.\s*/, '')}</li>)}
                </ol>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <h3 style={S.h3}>V. Patient-Facing Summary</h3>
                <p style={S.p}>{result.report.patient_summary}</p>
            </div>

            <div style={{ marginBottom: '20px', border: '1px solid #999', padding: '10px', background: '#fafafa' }}>
                <h3 style={{ ...S.h3, border: 'none', marginBottom: '6px' }}>VI. Stated Limitations</h3>
                <p style={{ fontSize: '11px', margin: 0, textAlign: 'justify' }}>{result.report.limitations}</p>
            </div>

            <div style={{ marginTop: '34px', display: 'flex', justifyContent: 'flex-end', pageBreakInside: 'avoid' }}>
                <div style={{ width: '300px', textAlign: 'center' }}>
                    <div style={{ borderBottom: '1px solid black', height: '38px', marginBottom: '4px' }} />
                    <p style={{ fontSize: '11.5px', margin: 0 }}>Reviewing Clinician — signature &amp; date</p>
                    <p style={{ fontSize: '10.5px', margin: 0, fontStyle: 'italic' }}>Report is not valid without countersignature</p>
                </div>
            </div>

            <div style={{ marginTop: '30px', borderTop: '1px solid #ccc', paddingTop: '9px', textAlign: 'center', fontSize: '9.5px', color: '#555' }}>
                <p style={{ margin: '0 0 3px 0' }}>
                    Generated by ClinicalVision AI ({model.architecture} @ {model.input_size}px, T={model.temperature?.toFixed(2)}) on {date} at {time}.
                </p>
                <p style={{ margin: 0 }}>
                    {result.disclaimer || 'Research prototype for clinical decision support only. Not a certified medical device.'}
                </p>
            </div>
        </div>
    );
});

PrintableReport.displayName = 'PrintableReport';
export default PrintableReport;
