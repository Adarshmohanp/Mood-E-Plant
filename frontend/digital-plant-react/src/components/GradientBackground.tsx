import React from 'react';
import styled, { keyframes } from 'styled-components';

interface Props {
    emotion: string;
}

const getGradientColors = (emotion: string): string[] => {
    switch (emotion.toLowerCase()) {
        case 'happy':
            return ['#FFD700', '#FFA500', '#FF8C00']; // Gold to Orange
        case 'sad':
            return ['#4682B4', '#000080', '#191970']; // Blue to Dark Blue
        case 'angry':
            return ['#FF0000', '#8B0000', '#800000']; // Red to Dark Red
        case 'neutral':
            return ['#98FB98', '#90EE90', '#3CB371']; // Light Green to Medium Green
        default:
            return ['#808080', '#696969', '#A9A9A9']; // Gray gradients
    }
};

const gradientAnimation = keyframes`
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
`;

const GradientContainer = styled.div<{ colors: string[] }>`
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: -1;
    background: linear-gradient(
        45deg,
        ${props => props.colors.join(', ')}
    );
    background-size: 400% 400%;
    animation: ${gradientAnimation} 15s ease infinite;
`;

const GradientBackground: React.FC<Props> = ({ emotion }) => {
    const colors = getGradientColors(emotion);
    return <GradientContainer colors={colors} />;
};

export default GradientBackground;