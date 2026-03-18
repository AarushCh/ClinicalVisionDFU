import React, { forwardRef } from 'react';

const PrintableReport = forwardRef(({ result, reportId }: any, ref: any) => {
    if (!result) return null;

    const date = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    const id = reportId || Math.floor(Math.random() * 100000);

    return (
        <div
            ref={ref}
            className="p-12 max-w-[850px] mx-auto"
            style={{
                fontFamily: '"Times New Roman", Times, serif', // Formal medical document font
                color: '#000000',
                backgroundColor: '#FFFFFF',
                WebkitPrintColorAdjust: 'exact',
                printColorAdjust: 'exact',
                lineHeight: '1.6'
            }}
        >
            {/* HEADER*/}
            <div style={{ borderBottom: '2px solid black', paddingBottom: '10px', marginBottom: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                <div>
                    <h1 style={{ fontSize: '28px', fontWeight: 'bold', margin: '0 0 5px 0' }}>ClinicalVision Diagnostic Laboratory</h1>
                    <p style={{ fontSize: '12px', fontStyle: 'italic', margin: 0 }}>Automated Neural Assessment & Triage System</p>
                </div>
                <div style={{ textAlign: 'right', fontSize: '12px' }}>
                    <p style={{ margin: '0 0 2px 0' }}><strong>Record ID:</strong> DFU-{id}</p>
                    <p style={{ margin: '0 0 2px 0' }}><strong>Date of Examination:</strong> {date}</p>
                    <p style={{ margin: 0 }}><strong>Time:</strong> {time}</p>
                </div>
            </div>

            {/* RISK DETERMINATION */}
            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                <h2 style={{ fontSize: '20px', textDecoration: 'underline', marginBottom: '10px' }}>DIAGNOSTIC REPORT: OPTICAL PLANTAR SCAN</h2>
                <div style={{
                    padding: '10px',
                    border: '2px solid black',
                    display: 'inline-block',
                    fontWeight: 'bold',
                    fontSize: '18px',
                    backgroundColor: '#f9f9f9'
                }}>
                    SYSTEM CLASSIFICATION: {result.risk} RISK ({(result.confidence * 100).toFixed(1)}% Confidence)
                </div>
            </div>

            {/* CLINICAL NARRATIVE (Written Paragraphs) */}
            <div style={{ marginBottom: '40px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 'bold', borderBottom: '1px solid black', marginBottom: '15px' }}>
                    I. Clinical Narrative & Assessment
                </h3>
                <p style={{ fontSize: '14px', marginBottom: '15px', textIndent: '30px', textAlign: 'justify' }}>
                    The ClinicalVision multi-modal AI engine has completed an automated analysis of the patient's plantar radiograph/optical scan. The neural network fused the visual pathology with the patient's submitted clinical history parameters. {result.report.clinical_assessment}
                </p>
                <p style={{ fontSize: '14px', marginBottom: '15px', textIndent: '30px', textAlign: 'justify' }}>
                    Based on the spatial analysis of the tissue morphology, the Convolutional Neural Network (ResNet-18) yielded the following pathological observation: {result.report.visual_analysis}
                </p>
            </div>

            {/* XAI VISUAL EVIDENCE */}
            <div style={{ marginBottom: '40px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 'bold', borderBottom: '1px solid black', marginBottom: '20px' }}>
                    II. Visual Evidence (Grad-CAM) & Feature Influence
                </h3>

                <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
                    {/* Image */}
                    <div style={{ flex: '0 0 40%' }}>
                        <div style={{ border: '2px solid black', padding: '4px', backgroundColor: '#fff' }}>
                            <img
                                src={`data:image/jpeg;base64,${result.heatmap}`}
                                alt="Radiograph Heatmap"
                                style={{ width: '100%', height: 'auto', display: 'block' }}
                            />
                        </div>
                        <p style={{ textAlign: 'center', fontSize: '11px', fontStyle: 'italic', marginTop: '8px' }}>
                            Fig 1: CNN Attention Gradients (Red = High Activation)
                        </p>
                    </div>

                    {/* SHAP Values Data */}
                    <div style={{ flex: '1' }}>
                        <p style={{ fontSize: '14px', marginBottom: '15px' }}>
                            The algorithmic decision was mathematically weighted based on the following patient vectors (SHAP Values):
                        </p>
                        <ul style={{ fontSize: '14px', margin: 0, paddingLeft: '20px' }}>
                            {result.shap?.map((s: any, i: number) => (
                                <li key={i} style={{ marginBottom: '8px' }}>
                                    <strong>{s.feature}:</strong> Contributed {s.value}% to the final diagnostic score.
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>

            {/* RECOMMENDED TRIAGE */}
            <div style={{ marginBottom: '40px', pageBreakBefore: 'always', pageBreakInside: 'avoid' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 'bold', borderBottom: '1px solid black', marginBottom: '15px' }}>
                    III. Recommended Triage & Action Plan
                </h3>
                <p style={{ fontSize: '14px', marginBottom: '10px' }}>
                    Pursuant to the {result.risk} risk classification, the system recommends the following clinical protocol:
                </p>
                <ol style={{ fontSize: '14px', paddingLeft: '30px', margin: 0 }}>
                    {result.report.triage.split('\n').map((step: string, index: number) => (
                        <li key={index} style={{ marginBottom: '8px' }}>
                            {step.replace(/^\d+\.\s*/, '')}
                        </li>
                    ))}
                </ol>
            </div>

            {/* FORMAL SIGN OFF */}
            <div style={{ marginTop: '50px', display: 'flex', justifyContent: 'flex-end', pageBreakInside: 'avoid' }}>
                <div style={{ width: '300px', textAlign: 'center' }}>
                    <div style={{ borderBottom: '1px solid black', height: '40px', marginBottom: '5px' }}></div>
                    <p style={{ fontSize: '12px', margin: 0 }}>Electronic Signature / Stamp</p>
                    <p style={{ fontSize: '12px', margin: 0, fontWeight: 'bold' }}>Attending Physician</p>
                </div>
            </div>

            {/* FOOTER */}
            <div style={{ marginTop: '40px', borderTop: '1px solid #ccc', paddingTop: '10px', textAlign: 'center', fontSize: '10px', color: '#666' }}>
                <p style={{ margin: '0 0 3px 0' }}>This report was automatically generated by the ClinicalVision Artificial Intelligence System.</p>
                <p style={{ margin: 0 }}>This document is intended for Clinical Decision Support (CDSS) only. Final diagnostic authority remains with the physician.</p>
            </div>

        </div>
    );
});

PrintableReport.displayName = 'PrintableReport';
export default PrintableReport;