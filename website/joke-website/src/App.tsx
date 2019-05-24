import React from 'react'
import './App.css'
import JokeGenerator from 'components/JokeGenerator/JokeGenerator'
import Blog from 'components/Blog/Blog'

const App: React.FC = () => {
  return (
    <div className="App" style={{
      backgroundImage: `url(${process.env.PUBLIC_URL}/laughing_emoji_64.png)`
    }}>
      <JokeGenerator />
      <Blog />
    </div>
  );
}

export default App
