import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

const TestPage = () => {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const res = await fetch('http://localhost:5000/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });
            const data = await res.json();
            if (res.ok) {
                setResponse(data.response);
            } else {
                setError(data.error || 'An error occurred');
            }
        } catch (err) {
            setError('Failed to fetch response');
        }
    };

    const handleSave = async () => {
        try {
            const res = await fetch('http://localhost:5000/save', { method: 'POST' });
            const data = await res.json();
            if (res.ok) {
                alert(data.message);
            } else {
                setError(data.error || 'Failed to save');
            }
        } catch (err) {
            setError('Failed to save');
        }
    };

    const handleDownload = async () => {
        try {
            const res = await fetch('http://localhost:5000/download');
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'vectorstore.faiss';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            } else {
                const data = await res.json();
                setError(data.error || 'Failed to download');
            }
        } catch (err) {
            setError('Failed to download');
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
            <div className="bg-white bg-opacity-20 p-8 rounded-lg shadow-lg w-full max-w-2xl">
                <h2 className="text-3xl font-bold mb-6 text-center">Test Your fine-tuned LLM</h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="w-full bg-white bg-opacity-50 p-2 rounded-md text-black"
                        rows="4"
                        placeholder="Enter your query here"
                    ></textarea>
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="w-full bg-red-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-md transition duration-300 ease-in-out"
                        type="submit"
                    >
                        Submit Query
                    </motion.button>
                </form>
                {response && (
                    <div className="mt-6 bg-white bg-opacity-30 p-4 rounded-md">
                        <h3 className="font-bold mb-2">Response:</h3>
                        <ReactMarkdown>{response}</ReactMarkdown>
                    </div>
                )}
                {error && <p className="text-red-500 text-center mt-4">{error}</p>}
                <div className="flex space-x-4 mt-6">
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex-1 bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-md transition duration-300 ease-in-out"
                        onClick={handleSave}
                    >
                        Save to Firebase
                    </motion.button>
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-md transition duration-300 ease-in-out"
                        onClick={handleDownload}
                    >
                        Download Vector DB
                    </motion.button>
                </div>
                <div className="mt-6 text-center">
                    <Link to="/upload" className="text-purple-300 hover:underline">Back to Upload</Link>
                </div>
            </div>
        </div>
    );
};

export default TestPage;