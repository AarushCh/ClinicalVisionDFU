"use client";
import { useState } from "react";
import axios from "axios";

export default function UploadForm({ setResult, setLoading, loading }: any) {
    const [file, setFile] = useState<File | null>(null);
    const [formData, setFormData] = useState({ age: "", bmi: "", diabetes_years: "" });

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) setFile(e.target.files[0]);
    };
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const onSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;
        setLoading(true);
        const data = new FormData();
        data.append("image", file);
        data.append("age", formData.age);
        data.append("bmi", formData.bmi);
        data.append("diabetes_years", formData.diabetes_years);

        try {
            const res = await axios.post("http://127.0.0.1:10000/predict", data);
            setResult(res.data);
        } catch (error) {
            alert("Backend connection failed.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-[#1e293b]/50 backdrop-blur-xl p-8 rounded-3xl border border-white/10 shadow-2xl">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                Patient Input
            </h2>
            <form onSubmit={onSubmit} className="space-y-5">

                <div className="relative group cursor-pointer">
                    <div className="absolute inset-0 bg-blue-500/20 rounded-xl blur group-hover:bg-blue-500/30 transition-all"></div>
                    <div className="relative border-2 border-dashed border-slate-600 rounded-xl p-6 text-center hover:border-blue-400 transition-colors bg-[#0f172a]">
                        <input type="file" onChange={handleFileChange} accept="image/*" className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" required />
                        <svg className="mx-auto h-10 w-10 text-slate-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                        <p className="text-sm font-medium text-slate-300">{file ? file.name : "Drag & Drop Plantar Image"}</p>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Age</label>
                        <input type="number" name="age" onChange={handleInputChange} className="w-full bg-[#0f172a] border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all" required placeholder="e.g. 55" />
                    </div>
                    <div className="space-y-1">
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">BMI</label>
                        <input type="number" step="0.1" name="bmi" onChange={handleInputChange} className="w-full bg-[#0f172a] border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all" required placeholder="e.g. 28.5" />
                    </div>
                </div>
                <div className="space-y-1">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Diabetes Duration (Years)</label>
                    <input type="number" name="diabetes_years" onChange={handleInputChange} className="w-full bg-[#0f172a] border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500 outline-none transition-all" required placeholder="e.g. 10" />
                </div>

                <button type="submit" disabled={loading} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-4 px-4 rounded-xl transition-all shadow-[0_0_20px_rgba(37,99,235,0.3)] disabled:opacity-50 mt-4">
                    {loading ? "Processing Neural Fusion..." : "Run AI Analysis"}
                </button>
            </form>
        </div>
    );
}