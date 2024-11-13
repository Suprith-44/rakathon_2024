import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const models = [
    { name: 'Cohere', image: require('./cohere.png') },
    { name: 'OpenAI', image: require('./openai.jpg') },
    { name: 'Llama', image: require('./llama.png') },
    { name: 'Gemini', image: require('./gemini.jpg') }
];

const ChooseLLMPage = () => {
    const navigate = useNavigate();

    const handleModelClick = (modelName) => {
        // Only navigate to the UploadPage if the model is "Cohere" or "Gemini"
        if (modelName === 'Cohere' || modelName === 'Gemini') {
            navigate('/upload');
        } else {
            // You might want to show an alert or handle other cases here
            alert('This model does not have an upload page.');
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
            <div className="bg-white bg-opacity-20 p-8 rounded-lg shadow-lg w-full max-w-md">
                <h2 className="text-3xl font-bold mb-6 text-center">Choose Your Base LLM Model</h2>
                <div className="grid grid-cols-2 gap-4">
                    {models.map((model) => (
                        <motion.div
                            key={model.name}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className="flex flex-col items-center bg-white bg-opacity-10 p-4 rounded-lg shadow-md cursor-pointer"
                            onClick={() => handleModelClick(model.name)}
                        >
                            <img src={model.image} alt={model.name} className="w-24 h-24 object-cover mb-2" />
                            <p className="text-lg font-bold">{model.name}</p>
                        </motion.div>
                    ))}
                </div>
                <div className="mt-6 text-center">
                    <Link to="/dashboard" className="text-white-500 hover:underline">Back to Dashboard</Link>
                </div>
            </div>
        </div>
    );
};

export default ChooseLLMPage;
