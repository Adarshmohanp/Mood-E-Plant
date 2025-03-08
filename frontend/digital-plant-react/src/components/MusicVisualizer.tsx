import React from 'react';
import styled, { keyframes } from 'styled-components';

interface Props {
    emotion: string;
}

const getBars = (count: number) => Array.from({ length: count }, (_, i) => i);

const getBarAnimation = (height: number) => keyframes`
    0%, 100% {
        height: 10px;
    }
    50% {
        height: ${height}px;
    }
`;

const Container = styled.div`
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 100px;
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 5px;
    padding: 20px;
    background: linear-gradient(to top, rgba(0,0,0,0.3), transparent);
`;

const Bar = styled.div<{ delay: number; emotion: string; height: number }>`
    width: 4px;
    height: 10px;
    border-radius: 2px;
    background-color: ${({ emotion }) => {
        switch (emotion.toLowerCase()) {
            case 'happy': return '#FFD700';
            case 'sad': return '#4682B4';
            case 'angry': return '#FF4444';
            case 'neutral': return '#98FB98';
            default: return '#FFFFFF';
        }
    }};
    animation: ${({ height }) => getBarAnimation(height)} ${({ delay }) => 0.5 + delay * 0.1}s ease-in-out infinite;
`;

const MusicInfo = styled.div`
    position: absolute;
    top: -30px;
    color: white;
    font-size: 14px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
`;

const MusicVisualizer: React.FC<Props> = ({ emotion }) => {
    const getRandomHeight = () => Math.floor(Math.random() * 40) + 20;
    
    return (
        <Container>
            <MusicInfo>Now Playing: {emotion} Mood Music</MusicInfo>
            {getBars(20).map((i) => (
                <Bar 
                    key={i} 
                    delay={i} 
                    emotion={emotion}
                    height={getRandomHeight()}
                />
            ))}
        </Container>
    );
};

export default MusicVisualizer;