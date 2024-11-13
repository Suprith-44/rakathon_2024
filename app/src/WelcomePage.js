import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Bot, Brain, Zap, ArrowRight } from 'lucide-react';

const WelcomePage = () => {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
      <motion.div
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <h3 className="text-6xl font-bold mb-4">RagAndUnlearn.ai</h3>
        <p className="text-2xl mb-8">Craft Dynamic, Intelligent Chatbots for Your Organization</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="flex justify-center space-x-8 mb-12"
      >
        <FeatureCard
          icon={<Bot size={40} />}
          title="Custom Chatbots"
          description="Tailor-made for your specific needs"
        />
        <FeatureCard
          icon={<Brain size={40} />}
          title="Real-time Learning"
          description="Adapt on the fly with RAG technology"
        />
        <FeatureCard
          icon={<Zap size={40} />}
          title="Machine Unlearning"
          description="Remove unwanted data effortlessly"
        />
      </motion.div>

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className="bg-white text-purple-600 font-bold py-3 px-6 rounded-full text-xl shadow-lg flex items-center"
        onClick={() => {
          // Navigate to login page
          navigate('/login')
        }}
      >
        Get Started
        <ArrowRight className="ml-2" />
      </motion.button>
    </div>
  );
};

const FeatureCard = ({ icon, title, description }) => (
  <motion.div
    whileHover={{ y: -5 }}
    className="bg-white bg-opacity-20 p-6 rounded-lg shadow-lg text-center w-64"
  >
    <div className="text-4xl mb-4 flex justify-center">{icon}</div>
    <h3 className="text-xl font-semibold mb-2">{title}</h3>
    <p className="text-sm">{description}</p>
  </motion.div>
);

export default WelcomePage;