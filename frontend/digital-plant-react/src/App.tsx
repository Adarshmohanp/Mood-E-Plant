import React, { useEffect, useState } from 'react';
import { fetchEmotion } from './services/emotionService';
import EmotionDisplay from './components/EmotionDisplay';
import Plant from './components/Plant';
import './App.css';

const App: React.FC = () => {
    const [emotion, setEmotion] = useState<string>('');
    const [plantImage, setPlantImage] = useState<string>('');

    useEffect(() => {
        const updateEmotion = async () => {
            try {
                const data = await fetchEmotion();
                setEmotion(data.emotion);
                setPlantImage(data.plant_image);
            } catch (error) {
                console.error('Error fetching emotion:', error);
            }
        };

        const interval = setInterval(updateEmotion, 1000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="App">
            <EmotionDisplay emotion={emotion} />
            <Plant plantImage={plantImage} />
        </div>
    );
};

export default App;
