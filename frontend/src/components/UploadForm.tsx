"use client";

import { useState } from "react";
import axios from "axios";

interface UploadFormProps {
    setResult: (data: any) => void;
    setLoading: (loading: boolean) => void;
    loading: boolean;
}

export default function UploadForm({ setResult, setLoading, loading }: UploadFormProps) {
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
            alert("Error connecting to AI Backend.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
            <h2 className="text-xl font-bold mb-4">Patient Data</h2>
            <form onSubmit={onSubmit} className="space-y-4">
                <div>
                    <label className="block text-sm font-medium mb-1">Foot Image</label>
                    <input type="file" onChange={handleFileChange} accept="image/*" className="w-full border rounded-lg p-2" required />
                </div>
                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label className="block text-sm font-medium mb-1">Age</label>
                        <input type="number" name="age" onChange={handleInputChange} className="w-full border rounded-lg p-2" required />
                    </div>
                    <div>
                        <label className="block text-sm font-medium mb-1">BMI</label>
                        <input type="number" step="0.1" name="bmi" onChange={handleInputChange} className="w-full border rounded-lg p-2" required />
                    </div>
                    <div>
                        <label className="block text-sm font-medium mb-1">Diabetes (Yrs)</label>
                        <input type="number" name="diabetes_years" onChange={handleInputChange} className="w-full border rounded-lg p-2" required />
                    </div>
                </div>
                <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg disabled:bg-blue-300"
                >
                    {loading ? "Analyzing..." : "Analyze Patient"}
                </button>
            </form>
        </div>
    );
}