import React, { useState } from 'react'
import Typist from 'react-typist'
import { Button, ButtonGroup } from 'reactstrap'
import { GridLoader } from 'react-spinners'
import Slider from 'rc-slider'
import 'rc-slider/assets/index.css'
import 'App.css'
import JokeUtil from 'utils/jokeUtils'

const GenerateJokeButton: React.FC<{
  getJoke: Function,
  setIsLoading: Function
}> = (props) => {
  return (
    <div className='generate-joke-button-container'>
      <Button
        color='primary'
        onClick={ async () => await props.getJoke() }>
        Generate Joke
      </Button>
    </div>

  )
}

const JokeDisplay: React.FC<{ jokeText: string }> = ({ jokeText }) => {
  return (
    <div className='joke-display'>
      <Typist>
        { jokeText }
      </Typist>
    </div>
  )
}

const JokeGenerator: React.FC = () => {
  const [temperature, setTemperature] = useState(0.5)
  const [maxNumberOfCharacters, setMaxChars] = useState(100)
  const [tokenType, setTokenType] = useState('character')
  const [jokeText, setJokeText] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const getJoke = async () => {
    setIsLoading(true)
    const jokeUtil = new JokeUtil()
    const joke = await jokeUtil.fetchJoke(tokenType, temperature, maxNumberOfCharacters)
    setJokeText(joke)
    setIsLoading(false)
  }

  const renderJokeContent = () => {
    if (isLoading) {
      return (
        <div className='generate-joke-loading-container'>
          <GridLoader
            color={ '#33dd88' }
            loading={ isLoading }
          />
        </div>
      )
    }

    if (jokeText) {
      return (
        <div>
          <JokeDisplay jokeText={ jokeText }/>
          <GenerateJokeButton
            setIsLoading={ setIsLoading }
            getJoke={ getJoke } />
        </div>
      )
    }

    return (
      <GenerateJokeButton
        setIsLoading={ setIsLoading }
        getJoke={ getJoke } />
    )
  }
  
  return (
    <div className="container joke-generator">
      <h1>
        <span role='img' aria-label='laughing emoji'>🤣</span>
        JokeBot
        <span role='img' aria-label='laughing emoji'>🤣</span>
      </h1>
      <div className='joke-display-container'>
        { renderJokeContent() }
        <div className='joke-options-container'>
          <div className='joke-option-container'>
            <div>
              Token Type:
            </div>
            <ButtonGroup>
              <Button
                color={ tokenType === 'character' ? 'primary' : 'secondary' }
                onClick={ () => setTokenType('character') }>
                Character
              </Button>
              <Button
                color={ tokenType === 'word' ? 'primary' : 'secondary' }
                onClick={ () => setTokenType('word') }>
                Word
              </Button>
            </ButtonGroup>
          </div>
          <div className='joke-option-container'>
            Temperature: { temperature }
            <Slider
              step={ 0.01 }
              min={ 0.0 }
              max={ 1.0 }
              defaultValue={ 0.5 }
              onChange={ value => setTemperature(value) }
            />
          </div>
          <div className='joke-option-container'>
            Maximum number of tokens to generate: { maxNumberOfCharacters }
            <Slider
              min={ 50 }
              max={ 500 }
              defaultValue={ 100 }
              onChange={ value => setMaxChars(value) }
            />
          </div>
        </div>
      </div>
      <div className='joke-disclaimer'>
        <h4>Disclaimer</h4>
        These jokes are automatically generated from various datasets (see below for dataset and model details).
        We do not intend for any of these jokes to be harmful and sincerely hope that is not the case.
        We apologize if any of these jokes are offensive.
      </div>
    </div>
  );
}

export default JokeGenerator