import axios from 'axios'

class JokeUtil {
    fetchJoke = async (tokenType: string, temperature: number, maxNumberCharacters: number) : Promise<string> => {
        // return Promise.resolve('What good is that when you get some life to the professor.')
        const response = await axios.get('http://localhost:5000')
        return response.data
    }
}

export default JokeUtil