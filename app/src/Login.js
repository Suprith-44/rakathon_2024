import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Lock, Mail } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const LoginPage = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch('http://localhost:5000/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });
    
            const result = await response.json();
            if (response.ok && result.success) {
                // Store the user's details in local storage
                localStorage.setItem('user', JSON.stringify(result.user));
    
                // Redirect to dashboard page upon successful login
                navigate('/dashboard');
            } else {
                setError(result.message || 'An error occurred while logging in.');
            }
        } catch (error) {
            setError('An error occurred while logging in.');
        }
    };
    

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
            <motion.div
                initial={{ opacity: 0, y: -50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="bg-white bg-opacity-20 p-8 rounded-lg shadow-lg w-full max-w-md"
            >
                <h2 className="text-3xl font-bold mb-6 text-center">Login to RagandUnlearn.ai</h2>
                <form onSubmit={handleLogin} className="space-y-4">
                    <div>
                        <label htmlFor="email" className="block mb-1">Email</label>
                        <div className="relative">
                            <input
                                type="email"
                                id="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="w-full px-3 py-2 bg-white bg-opacity-50 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-300 pl-10"
                                required
                            />
                            <Mail className="absolute left-3 top-2.5 text-gray-500" size={20} />
                        </div>
                    </div>
                    <div>
                        <label htmlFor="password" className="block mb-1">Password</label>
                        <div className="relative">
                            <input
                                type="password"
                                id="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full px-3 py-2 bg-white bg-opacity-50 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-300 pl-10"
                                required
                            />
                            <Lock className="absolute left-3 top-2.5 text-gray-500" size={20} />
                        </div>
                    </div>
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="w-full bg-red-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-md transition duration-300 ease-in-out"
                        type="submit"
                    >
                        Login
                    </motion.button>
                    {error && <p className="text-red-500 text-center mt-2">{error}</p>}
                </form>
                <div className="mt-4 text-center">
                    <a href="#" className="text-sm hover:underline">Forgot password?</a>
                </div>
                <div className="mt-6 text-center">
                    <p>Don't have an account? <a href="/signup" className="text-red-600 hover:underline">Sign up</a></p>
                </div>
            </motion.div>
            <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="mt-8 text-white flex items-center"
                onClick={() => navigate('/')}
            >
                <ArrowLeft className="mr-2" />
                Back to Home
            </motion.button>
        </div>
    );
};

export default LoginPage;
