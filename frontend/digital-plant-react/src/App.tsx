import React, { useEffect, useState } from 'react';
import EmotionDetector from './components/EmotionDetector';
import './App.css';

const App: React.FC = () => {
    return (
        <div className="App">
            <EmotionDetector />
        </div>
    );
};

export default App;
