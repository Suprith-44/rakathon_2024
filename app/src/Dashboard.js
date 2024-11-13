import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const DashboardPage = () => {
    const [user, setUser] = useState(null);
    const [llms, setLlms] = useState([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        // Retrieve the user's information from local storage
        const loggedInUser = localStorage.getItem('user');
        if (loggedInUser) {
            const parsedUser = JSON.parse(loggedInUser);
            setUser(parsedUser);
            // Fetch user's LLMs
            fetchUserLlms(parsedUser.email);
        } else {
            // If no user is logged in, redirect to the login page
            navigate('/login');
        }
    }, [navigate]);

    const fetchUserLlms = async (email) => {
        try {
            const response = await fetch(`http://localhost:5000/get-user-llms?email=${encodeURIComponent(email)}`);
            const data = await response.json();
            
            if (response.ok) {
                setLlms(data.llms);
            } else {
                console.error('Failed to fetch LLMs:', data.error);
            }
        } catch (error) {
            console.error('Error fetching LLMs:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        localStorage.removeItem('user');
        navigate('/login');
    };

    const handleLlmClick = (llmId) => {
        // Navigate to the LLM chat interface or details page
        navigate(`/llm/${llmId}`);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
            <div className="bg-white bg-opacity-20 p-8 rounded-lg shadow-lg w-full max-w-md">
                <h2 className="text-3xl font-bold mb-6 text-center">Dashboard</h2>
                <div className="space-y-4">
                    <Link to="/choosellm" className="block bg-red-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-md text-center transition duration-300 ease-in-out">
                        Create new AI assistant
                    </Link>
                    
                    <div className="text-center text-white">
                        <p className="text-xl font-semibold mb-4">Your AI assistants:</p>
                        {loading ? (
                            <p>Loading your AI assistants...</p>
                        ) : llms.length > 0 ? (
                            <div className="space-y-2">
                                {llms.map((llm) => (
                                    <button
                                        key={llm.id}
                                        onClick={() => handleLlmClick(llm.id)}
                                        className="w-full bg-white bg-opacity-10 hover:bg-opacity-20 py-2 px-4 rounded-md transition duration-300 ease-in-out"
                                    >
                                        {llm.llmName}
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <p>No fine-tuned LLMs yet. Create one to get started!</p>
                        )}
                    </div>
                </div>
                
                <div className="absolute top-4 right-4 text-right">
                    {user && (
                        <>
                            <button className="mt-2 bg-blue-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-md transition duration-300 ease-in-out">
                                {user.name}
                            </button>
                            <br />
                            <button
                                onClick={handleLogout}
                                className="mt-2 bg-red-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-md transition duration-300 ease-in-out"
                            >
                                Logout
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DashboardPage;