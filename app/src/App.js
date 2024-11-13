import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import WelcomePage from './WelcomePage';
import LoginPage from './Login';
import SignupPage from './SignUp';
import DashboardPage from './Dashboard';
import ChooseLLMPage from './choosellm';
import UploadPage from './UploadData';
import TestPage from './TestPage';
import QueryPage from './Query';
import ViewLLMPage from './viewllm';
import './index.css';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/choosellm" element={<ChooseLLMPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/test" element={<TestPage />} />
        <Route path="/query" element={<QueryPage />} />
        <Route path="/llm/:id" element={<ViewLLMPage />} />
        </Routes>
    </Router>
  );
};

export default App;
