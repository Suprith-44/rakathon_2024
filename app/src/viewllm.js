import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';

const ViewLLMPage = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [query, setQuery] = useState('');
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingLLM, setLoadingLLM] = useState(true);
  const [llmName, setLLMName] = useState('');
  const [user, setUser] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [unlearningText, setUnlearningText] = useState('');
  const [showUnlearnModal, setShowUnlearnModal] = useState(false);
  const [unlearning, setUnlearning] = useState(false);

  useEffect(() => {
    const userStr = localStorage.getItem('user');
    if (!userStr) {
      navigate('/login');
      return;
    }
    const userObj = JSON.parse(userStr);
    setUser(userObj);
    loadLLM();
  }, [navigate, id]);

  const loadLLM = async () => {
    try {
      const response = await fetch(`http://localhost:5000/load-llm/${id}`);
      const data = await response.json();
      
      if (response.ok) {
        setLLMName(data.llmName);
      } else {
        alert('Error loading LLM: ' + data.error);
        navigate('/dashboard');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error loading LLM');
      navigate('/dashboard');
    } finally {
      setLoadingLLM(false);
    }
  };

  const handleUnlearn = async () => {
    if (!unlearningText.trim()) return;
    
    setUnlearning(true);
    try {
      const response = await fetch('http://localhost:5000/unlearn-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: unlearningText,
          llmId: id,
          userEmail: user.email
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to unlearn data');
      }

      // Reload the LLM data
      await loadLLM();
      setShowUnlearnModal(false);
      setUnlearningText('');
      alert('Successfully unlearned the specified data');
    } catch (error) {
      console.error('Error:', error);
      alert('Error unlearning data');
    } finally {
      setUnlearning(false);
    }
  };

  const handleDownload = async (fileType) => {
    setDownloading(true);
    try {
      const response = await fetch(`http://localhost:5000/download-files/${fileType}`);
      
      if (!response.ok) {
        throw new Error('Download failed');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileType === 'faiss' ? 'faiss_index.bin' : 'texts.pkl';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading file:', error);
      alert('Error downloading file');
    } finally {
      setDownloading(false);
    }
  };

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

  if (loadingLLM) {
    return (
      <div className="max-w-4xl mx-auto p-5 font-sans">
        <div className="text-center">Loading your AI assistant...</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-5 font-sans">
      <h1 className="text-center text-2xl font-bold text-gray-800 mb-8">
        {llmName} - Ask Questions
      </h1>

      <div className="flex justify-between items-center mb-6">
        <Link 
          to="/dashboard"
          className="text-blue-600 hover:text-blue-800 transition-colors"
        >
          ← Back to Dashboard
        </Link>
        <div className="flex gap-3">
          <button
            onClick={() => setShowUnlearnModal(true)}
            className="px-4 py-2 rounded-md text-white bg-red-600 hover:bg-red-700 transition-colors"
          >
            Unlearn Data
          </button>
          <button
            onClick={() => handleDownload('faiss')}
            disabled={downloading}
            className={`px-4 py-2 rounded-md text-white transition-colors ${
              downloading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {downloading ? 'Downloading...' : 'Download FAISS Index'}
          </button>
          <button
            onClick={() => handleDownload('texts')}
            disabled={downloading}
            className={`px-4 py-2 rounded-md text-white transition-colors ${
              downloading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {downloading ? 'Downloading...' : 'Download Texts'}
          </button>
        </div>
      </div>

      {showUnlearnModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg w-96">
            <h2 className="text-xl font-bold mb-4">Unlearn Data</h2>
            <textarea
              value={unlearningText}
              onChange={(e) => setUnlearningText(e.target.value)}
              placeholder="Enter the text you want to unlearn..."
              className="w-full h-32 p-2 border rounded mb-4"
            />
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowUnlearnModal(false)}
                className="px-4 py-2 rounded-md text-gray-600 hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                onClick={handleUnlearn}
                disabled={unlearning || !unlearningText.trim()}
                className={`px-4 py-2 rounded-md text-white ${
                  unlearning || !unlearningText.trim()
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-red-600 hover:bg-red-700'
                }`}
              >
                {unlearning ? 'Unlearning...' : 'Unlearn'}
              </button>
            </div>
          </div>
        </div>
      )}

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
    </div>
  );
};

export default ViewLLMPage;