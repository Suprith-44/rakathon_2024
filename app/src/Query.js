import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const QueryPage = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [savingLLM, setSavingLLM] = useState(false);
  const [llmName, setLLMName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [user, setUser] = useState(null);

  useEffect(() => {
    const userStr = localStorage.getItem('user');
    if (!userStr) {
      navigate('/login');
      return;
    }
    const userObj = JSON.parse(userStr);
    setUser(userObj);
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query,
          history: conversations 
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setConversations(prev => [...prev, {
          question: query,
          answer: data.answer,
          timestamp: new Date().toISOString()
        }]);
        setQuery('');
      } else {
        setConversations(prev => [...prev, {
          question: query,
          answer: 'Error getting response',
          timestamp: new Date().toISOString()
        }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setConversations(prev => [...prev, {
        question: query,
        answer: 'Error connecting to server',
        timestamp: new Date().toISOString()
      }]);
    }
    setLoading(false);
  };

  const handleSaveLLM = async () => {
    if (!llmName.trim()) return;
    
    setSavingLLM(true);
    try {
      const response = await fetch('http://localhost:5000/save-llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          llmName: llmName,
          userEmail: user.email
        }),
      });

      if (response.ok) {
        alert('LLM saved successfully!');
        setShowSaveDialog(false);
        setLLMName('');
        navigate('/dashboard'); // Redirect to dashboard page after saving
      } else {
        alert('Error saving LLM');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error saving LLM');
    }
    setSavingLLM(false);
  };

  return (
    <div className="max-w-4xl mx-auto p-5 font-sans">
      <h1 className="text-center text-2xl font-bold text-gray-800 mb-8">
        ❓ Test Your context-specific AI assistant
      </h1>

      <div className="flex justify-between items-center mb-6">
        <Link 
          to="/upload"
          className="text-blue-600 hover:text-blue-800 transition-colors"
        >
          ← Back to Upload
        </Link>
        <button
          onClick={() => setShowSaveDialog(true)}
          className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition-colors"
        >
          Save AI assistant
        </button>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3 mb-8">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your question"
          className="flex-1 px-4 py-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className={`px-6 py-3 rounded-md text-white transition-colors ${
            loading || !query.trim() 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      <div className="h-[600px] overflow-y-auto border border-gray-200 rounded-md p-5">
        {conversations.length === 0 ? (
          <div className="text-center text-gray-500 py-10">
            No questions asked yet. Start by typing a question above!
          </div>
        ) : (
          conversations.map((conv, index) => (
            <div 
              key={index}
              className="bg-white rounded-lg p-5 mb-5 shadow-sm"
            >
              <div className="mb-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center font-bold text-blue-700">
                    Q
                  </div>
                  <div>
                    <p className="text-gray-800 mb-1">{conv.question}</p>
                    <p className="text-sm text-gray-500">
                      {new Date(conv.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center font-bold text-green-700">
                  A
                </div>
                <p className="text-gray-800 whitespace-pre-wrap">{conv.answer}</p>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Save LLM Modal */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96">
            <h3 className="text-xl font-semibold mb-4">Save Fine-tuned LLM</h3>
            <input
              type="text"
              value={llmName}
              onChange={(e) => setLLMName(e.target.value)}
              placeholder="Enter LLM name"
              className="w-full px-3 py-2 rounded-md border border-gray-300 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-100 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveLLM}
                disabled={savingLLM || !llmName.trim()}
                className={`px-4 py-2 rounded-md text-white ${
                  savingLLM || !llmName.trim()
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {savingLLM ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QueryPage;
