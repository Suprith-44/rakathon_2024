import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    setProcessing(true);
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setSuccess(true);
      } else {
        alert('Error processing PDFs');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error uploading files');
    }
    setProcessing(false);
  };

  return (
    <div style={{
      maxWidth: '800px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h1 style={{
        textAlign: 'center',
        color: '#333',
        marginBottom: '30px'
      }}> Create AI assistant</h1>

      <div style={{
        border: '2px dashed #ccc',
        padding: '20px',
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileChange}
          style={{ marginBottom: '20px' }}
        />
        
        <div style={{ marginBottom: '20px' }}>
          Selected files: {files.map(file => file.name).join(', ')}
        </div>

        <button
          onClick={handleUpload}
          disabled={processing || files.length === 0}
          style={{
            backgroundColor: '#007bff',
            color: 'white',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '4px',
            cursor: files.length === 0 ? 'not-allowed' : 'pointer'
          }}
        >
          {processing ? 'Processing...' : 'Process PDFs'}
        </button>
      </div>

      {success && (
        <div style={{
          backgroundColor: '#d4edda',
          color: '#155724',
          padding: '15px',
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          PDFs processed successfully!
          <button
            onClick={() => navigate('/query')}
            style={{
              backgroundColor: '#28a745',
              color: 'white',
              padding: '10px 20px',
              border: 'none',
              borderRadius: '4px',
              marginLeft: '10px',
              cursor: 'pointer'
            }}
          >
            Go to Query Page
          </button>
        </div>
      )}
    </div>
  );
};

export default UploadPage;